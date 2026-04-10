use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::{
    ChatResponse, ContentBlock, Error, ExtraMap, FinishReason, ImageSource, ResponseMetadata,
    Result, ToolCallRef, Usage,
};

/// Boxed stream of normalized chat response events
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send + 'static>>;

/// A zero-allocation stream that emits the normalized event sequence for a
/// fully materialized [`ChatResponse`].
pub struct SingleResponseStream {
    inner: ResponseStreamEventIter,
}

impl SingleResponseStream {
    /// Create a normalized stream adapter from a fully materialized response.
    #[must_use]
    pub fn new(response: ChatResponse) -> Self {
        Self {
            inner: response.stream_events(),
        }
    }
}

impl From<ChatResponse> for SingleResponseStream {
    fn from(response: ChatResponse) -> Self {
        Self::new(response)
    }
}

impl Stream for SingleResponseStream {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.inner.next().map(Ok))
    }
}

/// Interpretation mode for streaming usage metadata
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum UsageMetadataMode {
    /// Usage values are incremental and should be added to any prior usage.
    Delta,
    /// Usage values are cumulative snapshots and should replace prior usage.
    #[default]
    Snapshot,
}

fn is_default_usage_metadata_mode(mode: &UsageMetadataMode) -> bool {
    *mode == UsageMetadataMode::Snapshot
}

/// Tracks whether a stream was fully received without missing structural events.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCompleteness {
    /// Whether a `ResponseStart` event was observed.
    pub saw_response_start: bool,
    /// Whether a `ResponseStop` event was observed.
    pub saw_response_stop: bool,
    /// Whether any content block was left open when collection finished.
    pub has_open_blocks: bool,
    /// Whether partial tool calls had to be discarded during partial collection.
    pub dropped_incomplete_tool_calls: bool,
}

impl StreamCompleteness {
    /// Create an empty completeness tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            saw_response_start: false,
            saw_response_stop: false,
            has_open_blocks: false,
            dropped_incomplete_tool_calls: false,
        }
    }

    /// Return whether the stream was structurally complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.saw_response_start
            && self.saw_response_stop
            && !self.has_open_blocks
            && !self.dropped_incomplete_tool_calls
    }
}

/// The result of collecting a [`ChatStream`] into a [`ChatResponse`] via
/// [`StreamCollector`], paired with structural completeness information and
/// any terminal stream error that stopped collection early.
#[derive(Debug, Clone)]
pub struct CollectedResponse {
    /// Reconstructed response built from the received stream events.
    pub response: ChatResponse,
    /// Structural completeness information for the collected stream.
    pub completeness: StreamCompleteness,
    /// The terminal stream error that stopped collection early, if any.
    pub terminal_error: Option<Error>,
}

/// The type of content block being streamed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamBlockType {
    /// A text content block.
    Text,
    /// An image content block.
    Image,
    /// A chain-of-thought reasoning block.
    Reasoning,
    /// A tool/function call block.
    ToolCall,
    /// A provider-specific block type not covered by the standard variants.
    Other,
}

/// A single event in a streaming chat completion response.
///
/// Events typically arrive in order: `ResponseStart`, then one or more block
/// sequences (`BlockStart` / deltas / `BlockStop`), optionally
/// `ResponseMetadata`, and finally `ResponseStop`.
///
/// Normalized streams produced by this crate always emit exactly one
/// [`StreamEvent::ResponseMetadata`] before [`StreamEvent::ResponseStop`], even
/// when all response-level fields are absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamEvent {
    /// The response stream has started. Carries optional response-level identity.
    ResponseStart {
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Provider-assigned response identifier, when known at stream start.
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Provider-reported model identifier, when known at stream start.
        model: Option<String>,
    },
    /// A new content block has started at the given `index`.
    BlockStart {
        /// Stable block index used by later deltas and stop events.
        index: usize,
        /// Portable type of block being streamed.
        block_type: StreamBlockType,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Tool call identifier for tool-call blocks.
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Tool name for tool-call blocks.
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Provider-specific type name for [`StreamBlockType::Other`] blocks.
        type_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Block-start payload for block types that carry structured data up front.
        data: Option<ExtraMap>,
    },
    /// An incremental text chunk for the block at `index`.
    TextDelta {
        /// Block index this delta belongs to.
        index: usize,
        /// Incremental text payload.
        text: String,
    },
    /// An incremental reasoning chunk for the block at `index`.
    ReasoningDelta {
        /// Block index this delta belongs to.
        index: usize,
        /// Incremental reasoning text payload.
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Optional provider-specific signature associated with the reasoning block.
        signature: Option<String>,
    },
    /// An incremental tool call arguments chunk for the block at `index`.
    ToolCallDelta {
        /// Block index this delta belongs to.
        index: usize,
        /// Incremental JSON-string arguments payload.
        arguments: String,
    },
    /// The content block at `index` has finished.
    BlockStop {
        /// Block index being closed.
        index: usize,
    },
    /// Response-level metadata (usage, finish reason, identity). May arrive
    /// multiple times; consumers should merge/overwrite as appropriate.
    ResponseMetadata {
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Latest finish reason reported by the provider.
        finish_reason: Option<FinishReason>,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Latest usage payload reported by the provider.
        usage: Option<Usage>,
        /// Describes how to interpret this event's `usage` value.
        ///
        /// `Snapshot` means the event carries cumulative usage totals and should
        /// replace prior usage. `Delta` means the event carries incremental usage
        /// and should be added to prior usage.
        #[serde(default, skip_serializing_if = "is_default_usage_metadata_mode")]
        usage_mode: UsageMetadataMode,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Latest provider-assigned response identifier.
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        /// Latest provider-reported model identifier.
        model: Option<String>,
        #[serde(default, skip_serializing_if = "ExtraMap::is_empty")]
        /// Portable provider-specific metadata fields.
        metadata: ExtraMap,
    },
    /// The response stream has ended.
    ResponseStop,
}

/// Accumulates [`StreamEvent`]s into a [`CollectedResponse`].
///
/// Feed events via [`push`](Self::push), then call [`finish`](Self::finish)
/// or [`finish_partial`](Self::finish_partial) to produce the collected response.
///
/// Repeated [`StreamEvent::ResponseMetadata`] events are merged deterministically:
/// identity fields use the latest non-`None` value, portable metadata maps are
/// extended, and usage is either replaced or accumulated according to
/// `usage_mode` on each event.
#[derive(Default)]
pub struct StreamCollector {
    blocks: Vec<Option<BlockAccumulator>>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    model: Option<String>,
    id: Option<String>,
    metadata: ResponseMetadata,
    saw_response_start: bool,
    saw_response_stop: bool,
    terminal_error: Option<Error>,
}

impl StreamCollector {
    /// Create an empty stream collector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Processes a single owned stream event, accumulating it into the collector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Stream`] if a delta event targets a block with a
    /// mismatched type (e.g., a `TextDelta` for a `ToolCall` block).
    pub fn push(&mut self, event: StreamEvent) -> Result<()> {
        self.ensure_event_order(&event)?;

        match event {
            StreamEvent::ResponseStart { id, model } => {
                self.merge_response_start(id, model);
            }
            StreamEvent::BlockStart {
                index,
                block_type,
                id,
                name,
                type_name,
                data,
            } => self.start_block(index, block_type, id, name, type_name, data)?,
            StreamEvent::TextDelta { index, text } => {
                let block = self.ensure_text_block(index)?;
                if let BlockAccumulator::Text { text: buf, .. } = block {
                    append_or_replace(buf, text);
                }
            }
            StreamEvent::ReasoningDelta {
                index,
                text,
                signature,
            } => {
                let block = self.ensure_reasoning_block(index)?;
                if let BlockAccumulator::Reasoning {
                    text: buf,
                    signature: sig,
                    ..
                } = block
                {
                    append_or_replace(buf, text);
                    if let Some(signature) = signature {
                        *sig = Some(signature);
                    }
                }
            }
            StreamEvent::ToolCallDelta { index, arguments } => {
                let block = self.ensure_tool_call_block(index)?;
                if let BlockAccumulator::ToolCall { arguments: buf, .. } = block {
                    append_or_replace(buf, arguments);
                }
            }
            StreamEvent::BlockStop { index } => {
                self.close_block(index)?;
            }
            StreamEvent::ResponseMetadata {
                finish_reason,
                usage,
                usage_mode,
                id,
                model,
                metadata,
            } => {
                self.merge_response_metadata(finish_reason, usage, usage_mode, id, model, metadata)
            }
            StreamEvent::ResponseStop => {
                self.saw_response_stop = true;
            }
        }
        Ok(())
    }

    /// Processes a borrowed stream event.
    pub fn push_ref(&mut self, event: &StreamEvent) -> Result<()> {
        self.ensure_event_order(event)?;

        match event {
            StreamEvent::ResponseStart { id, model } => {
                self.merge_response_start(id.clone(), model.clone());
            }
            StreamEvent::BlockStart {
                index,
                block_type,
                id,
                name,
                type_name,
                data,
            } => self.start_block(
                *index,
                block_type.clone(),
                id.clone(),
                name.clone(),
                type_name.clone(),
                data.clone(),
            )?,
            StreamEvent::TextDelta { index, text } => {
                let block = self.ensure_text_block(*index)?;
                if let BlockAccumulator::Text { text: buf, .. } = block {
                    buf.push_str(text);
                }
            }
            StreamEvent::ReasoningDelta {
                index,
                text,
                signature,
            } => {
                let block = self.ensure_reasoning_block(*index)?;
                if let BlockAccumulator::Reasoning {
                    text: buf,
                    signature: sig,
                    ..
                } = block
                {
                    buf.push_str(text);
                    if let Some(signature) = signature {
                        *sig = Some(signature.clone());
                    }
                }
            }
            StreamEvent::ToolCallDelta { index, arguments } => {
                let block = self.ensure_tool_call_block(*index)?;
                if let BlockAccumulator::ToolCall { arguments: buf, .. } = block {
                    buf.push_str(arguments);
                }
            }
            StreamEvent::BlockStop { index } => {
                self.close_block(*index)?;
            }
            StreamEvent::ResponseMetadata {
                finish_reason,
                usage,
                usage_mode,
                id,
                model,
                metadata,
            } => self.merge_response_metadata(
                finish_reason.clone(),
                usage.clone(),
                *usage_mode,
                id.clone(),
                model.clone(),
                metadata.clone(),
            ),
            StreamEvent::ResponseStop => {
                self.saw_response_stop = true;
            }
        }

        Ok(())
    }

    pub(crate) fn set_terminal_error(&mut self, error: Error) {
        self.terminal_error = Some(error);
    }

    /// Finalizes the stream, requiring it to be structurally complete.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Stream`] if the stream is incomplete (missing
    /// `ResponseStart`/`ResponseStop`, has open blocks, or dropped incomplete
    /// tool calls). Also returns [`Error::Stream`] if tool call blocks are
    /// missing required `id` or `name` metadata.
    pub fn finish(self) -> Result<ChatResponse> {
        let collected = self.build_response(true)?;
        if !collected.completeness.is_complete() || collected.terminal_error.is_some() {
            return Err(Error::Stream(format!(
                "stream incomplete: start={}, stop={}, open_blocks={}, dropped_incomplete_tool_calls={}, terminal_error={}",
                collected.completeness.saw_response_start,
                collected.completeness.saw_response_stop,
                collected.completeness.has_open_blocks,
                collected.completeness.dropped_incomplete_tool_calls,
                collected
                    .terminal_error
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "none".to_owned()),
            )));
        }
        Ok(collected.response)
    }

    /// Finalizes a partial stream and returns the reconstructed response plus
    /// explicit completeness information.
    ///
    /// Unlike [`finish`](Self::finish), this does not error on incomplete
    /// streams. Incomplete tool calls are silently dropped and reported via
    /// [`StreamCompleteness`], and terminal provider/transport errors are
    /// preserved on [`CollectedResponse::terminal_error`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Stream`] only if internal invariants are violated
    /// (e.g., mismatched block types during reassembly).
    pub fn finish_partial(self) -> Result<CollectedResponse> {
        self.build_response(false)
    }

    /// Returns fully identified tool calls accumulated so far.
    ///
    /// This view includes only tool call blocks that currently have both `id`
    /// and `name`, even if the surrounding stream has not finished yet.
    pub fn tool_calls(&self) -> impl Iterator<Item = ToolCallRef<'_>> {
        self.blocks
            .iter()
            .filter_map(|block| match block.as_ref()? {
                BlockAccumulator::ToolCall {
                    id: Some(id),
                    name: Some(name),
                    arguments,
                    ..
                } => Some(ToolCallRef {
                    id,
                    name,
                    arguments,
                }),
                _ => None,
            })
    }

    fn ensure_event_order(&self, event: &StreamEvent) -> Result<()> {
        if !self.saw_response_stop {
            return Ok(());
        }

        let message = match event {
            StreamEvent::ResponseStop => "duplicate response_stop".to_owned(),
            _ => format!("{} received after response_stop", stream_event_name(event)),
        };

        Err(Error::Stream(message))
    }

    fn build_response(self, strict: bool) -> Result<CollectedResponse> {
        let mut content = Vec::with_capacity(self.blocks.len());
        let mut has_open_blocks = false;
        let mut dropped_incomplete_tool_calls = false;

        for block in self.blocks.into_iter().flatten() {
            match block {
                BlockAccumulator::Text { text, closed, .. } => {
                    if !closed {
                        has_open_blocks = true;
                    }
                    if !text.is_empty() {
                        content.push(ContentBlock::Text { text });
                    }
                }
                BlockAccumulator::Image { source, closed, .. } => {
                    if !closed {
                        has_open_blocks = true;
                    }
                    content.push(ContentBlock::Image { source });
                }
                BlockAccumulator::Reasoning {
                    text,
                    signature,
                    closed,
                    ..
                } => {
                    if !closed {
                        has_open_blocks = true;
                    }
                    if !text.is_empty() || signature.is_some() {
                        content.push(ContentBlock::Reasoning { text, signature });
                    }
                }
                BlockAccumulator::ToolCall {
                    id,
                    name,
                    arguments,
                    closed,
                    ..
                } => {
                    if !closed {
                        has_open_blocks = true;
                    }

                    match (id, name, closed) {
                        (Some(id), Some(name), true) => {
                            let arguments = if arguments.trim().is_empty() {
                                "{}".to_string()
                            } else {
                                arguments
                            };

                            if !is_valid_json_document(&arguments) {
                                if strict {
                                    return Err(Error::Stream(
                                        "tool call arguments were not valid JSON during collection"
                                            .into(),
                                    ));
                                }
                                dropped_incomplete_tool_calls = true;
                                continue;
                            }

                            content.push(ContentBlock::ToolCall {
                                id,
                                name,
                                arguments,
                            });
                        }
                        (Some(_), Some(_), false)
                        | (Some(_) | None, None, _)
                        | (None, Some(_), _) => {
                            if strict {
                                if !closed {
                                    return Err(Error::Stream(
                                        "tool call block remained open during collection".into(),
                                    ));
                                }
                                return Err(Error::Stream(
                                    "tool call block missing id or name during collection".into(),
                                ));
                            }
                            dropped_incomplete_tool_calls = true;
                        }
                    }
                }
                BlockAccumulator::Other {
                    type_name,
                    data,
                    closed,
                    ..
                } => {
                    if !closed {
                        has_open_blocks = true;
                    }
                    content.push(ContentBlock::Other { type_name, data });
                }
            }
        }

        Ok(CollectedResponse {
            response: ChatResponse {
                content,
                finish_reason: self.finish_reason,
                usage: self.usage,
                model: self.model,
                id: self.id,
                metadata: self.metadata,
            },
            completeness: StreamCompleteness {
                saw_response_start: self.saw_response_start,
                saw_response_stop: self.saw_response_stop,
                has_open_blocks,
                dropped_incomplete_tool_calls,
            },
            terminal_error: self.terminal_error,
        })
    }

    fn ensure_block_slot(&mut self, index: usize) {
        if index >= self.blocks.len() {
            self.blocks.resize_with(index + 1, || None);
        }
    }

    fn merge_response_start(&mut self, id: Option<String>, model: Option<String>) {
        self.saw_response_start = true;
        if let Some(id) = id {
            self.id = Some(id);
        }
        if let Some(model) = model {
            self.model = Some(model);
        }
    }

    fn merge_response_metadata(
        &mut self,
        finish_reason: Option<FinishReason>,
        usage: Option<Usage>,
        usage_mode: UsageMetadataMode,
        id: Option<String>,
        model: Option<String>,
        metadata: ExtraMap,
    ) {
        if let Some(reason) = finish_reason {
            self.finish_reason = Some(reason);
        }
        if let Some(chunk_usage) = usage {
            match usage_mode {
                UsageMetadataMode::Delta => match &mut self.usage {
                    Some(existing) => *existing += chunk_usage,
                    None => self.usage = Some(chunk_usage),
                },
                UsageMetadataMode::Snapshot => {
                    self.usage = Some(chunk_usage);
                }
            }
        }
        if let Some(id) = id {
            self.id = Some(id);
        }
        if let Some(model) = model {
            self.model = Some(model);
        }
        self.metadata.extend_portable(metadata);
    }

    fn start_block(
        &mut self,
        index: usize,
        block_type: StreamBlockType,
        id: Option<String>,
        name: Option<String>,
        type_name: Option<String>,
        data: Option<ExtraMap>,
    ) -> Result<()> {
        self.ensure_block_slot(index);
        let slot = &mut self.blocks[index];

        match block_type {
            StreamBlockType::Text => match slot {
                None => {
                    *slot = Some(BlockAccumulator::Text {
                        text: String::new(),
                        started: true,
                        closed: false,
                    });
                    Ok(())
                }
                Some(BlockAccumulator::Text {
                    started, closed, ..
                }) => mark_block_started(started, *closed, index, "text"),
                Some(_) => Err(Error::Stream(format!(
                    "block_start for text block conflicted with existing block at index {index}"
                ))),
            },
            StreamBlockType::Image => match slot {
                None => {
                    *slot = Some(BlockAccumulator::Image {
                        source: parse_image_source_from_block_data(data)?,
                        started: true,
                        closed: false,
                    });
                    Ok(())
                }
                Some(BlockAccumulator::Image {
                    started, closed, ..
                }) => mark_block_started(started, *closed, index, "image"),
                Some(_) => Err(Error::Stream(format!(
                    "block_start for image block conflicted with existing block at index {index}"
                ))),
            },
            StreamBlockType::Reasoning => match slot {
                None => {
                    *slot = Some(BlockAccumulator::Reasoning {
                        text: String::new(),
                        signature: None,
                        started: true,
                        closed: false,
                    });
                    Ok(())
                }
                Some(BlockAccumulator::Reasoning {
                    started, closed, ..
                }) => mark_block_started(started, *closed, index, "reasoning"),
                Some(_) => Err(Error::Stream(format!(
                    "block_start for reasoning block conflicted with existing block at index {index}"
                ))),
            },
            StreamBlockType::ToolCall => match slot {
                None => {
                    *slot = Some(BlockAccumulator::ToolCall {
                        id,
                        name,
                        arguments: String::new(),
                        started: true,
                        closed: false,
                    });
                    Ok(())
                }
                Some(BlockAccumulator::ToolCall {
                    id: existing_id,
                    name: existing_name,
                    started,
                    closed,
                    ..
                }) => {
                    mark_block_started(started, *closed, index, "tool_call")?;
                    merge_optional_field(existing_id, id, index, "tool call", "id")?;
                    merge_optional_field(existing_name, name, index, "tool call", "name")
                }
                Some(_) => Err(Error::Stream(format!(
                    "block_start for tool call conflicted with existing block at index {index}"
                ))),
            },
            StreamBlockType::Other => match slot {
                None => {
                    *slot = Some(BlockAccumulator::Other {
                        type_name: type_name.unwrap_or_else(|| "other".to_string()),
                        data: data.unwrap_or_default(),
                        started: true,
                        closed: false,
                    });
                    Ok(())
                }
                Some(BlockAccumulator::Other {
                    started, closed, ..
                }) => mark_block_started(started, *closed, index, "other"),
                Some(_) => Err(Error::Stream(format!(
                    "block_start for other block conflicted with existing block at index {index}"
                ))),
            },
        }
    }

    fn close_block(&mut self, index: usize) -> Result<()> {
        let block = self
            .blocks
            .get_mut(index)
            .and_then(Option::as_mut)
            .ok_or_else(|| Error::Stream(format!("block_stop for unknown index {index}")))?;
        match block {
            BlockAccumulator::Text { closed, .. } => mark_block_closed(closed, index, "text"),
            BlockAccumulator::Image { closed, .. } => mark_block_closed(closed, index, "image"),
            BlockAccumulator::Reasoning { closed, .. } => {
                mark_block_closed(closed, index, "reasoning")
            }
            BlockAccumulator::ToolCall { closed, .. } => {
                mark_block_closed(closed, index, "tool_call")
            }
            BlockAccumulator::Other { closed, .. } => mark_block_closed(closed, index, "other"),
        }
    }

    fn ensure_text_block(&mut self, index: usize) -> Result<&mut BlockAccumulator> {
        self.ensure_block_slot(index);
        let slot = &mut self.blocks[index];
        if slot.is_none() {
            *slot = Some(BlockAccumulator::Text {
                text: String::new(),
                started: false,
                closed: false,
            });
        }
        match slot.as_mut() {
            Some(block @ BlockAccumulator::Text { .. }) => Ok(block),
            _ => Err(Error::Stream(format!(
                "text delta received for non-text block at index {index}"
            ))),
        }
    }

    fn ensure_reasoning_block(&mut self, index: usize) -> Result<&mut BlockAccumulator> {
        self.ensure_block_slot(index);
        let slot = &mut self.blocks[index];
        if slot.is_none() {
            *slot = Some(BlockAccumulator::Reasoning {
                text: String::new(),
                signature: None,
                started: false,
                closed: false,
            });
        }
        match slot.as_mut() {
            Some(block @ BlockAccumulator::Reasoning { .. }) => Ok(block),
            _ => Err(Error::Stream(format!(
                "reasoning delta received for non-reasoning block at index {index}"
            ))),
        }
    }

    fn ensure_tool_call_block(&mut self, index: usize) -> Result<&mut BlockAccumulator> {
        self.ensure_block_slot(index);
        let slot = &mut self.blocks[index];
        if slot.is_none() {
            *slot = Some(BlockAccumulator::ToolCall {
                id: None,
                name: None,
                arguments: String::new(),
                started: false,
                closed: false,
            });
        }
        match slot.as_mut() {
            Some(block @ BlockAccumulator::ToolCall { .. }) => Ok(block),
            _ => Err(Error::Stream(format!(
                "tool call delta received for non-tool block at index {index}"
            ))),
        }
    }
}

enum BlockAccumulator {
    Text {
        text: String,
        started: bool,
        closed: bool,
    },
    Image {
        source: ImageSource,
        started: bool,
        closed: bool,
    },
    Reasoning {
        text: String,
        signature: Option<String>,
        started: bool,
        closed: bool,
    },
    ToolCall {
        id: Option<String>,
        name: Option<String>,
        arguments: String,
        started: bool,
        closed: bool,
    },
    Other {
        type_name: String,
        data: ExtraMap,
        started: bool,
        closed: bool,
    },
}

fn mark_block_started(
    started: &mut bool,
    closed: bool,
    index: usize,
    block_name: &str,
) -> Result<()> {
    if closed {
        return Err(Error::Stream(format!(
            "block_start for closed {block_name} block at index {index}"
        )));
    }
    if *started {
        return Err(Error::Stream(format!(
            "duplicate block_start for {block_name} block at index {index}"
        )));
    }
    *started = true;
    Ok(())
}

fn mark_block_closed(closed: &mut bool, index: usize, block_name: &str) -> Result<()> {
    if *closed {
        return Err(Error::Stream(format!(
            "duplicate block_stop for {block_name} block at index {index}"
        )));
    }
    *closed = true;
    Ok(())
}

fn stream_event_name(event: &StreamEvent) -> &'static str {
    match event {
        StreamEvent::ResponseStart { .. } => "response_start",
        StreamEvent::BlockStart { .. } => "block_start",
        StreamEvent::TextDelta { .. } => "text_delta",
        StreamEvent::ReasoningDelta { .. } => "reasoning_delta",
        StreamEvent::ToolCallDelta { .. } => "tool_call_delta",
        StreamEvent::BlockStop { .. } => "block_stop",
        StreamEvent::ResponseMetadata { .. } => "response_metadata",
        StreamEvent::ResponseStop => "response_stop",
    }
}

fn merge_optional_field(
    existing: &mut Option<String>,
    incoming: Option<String>,
    index: usize,
    block_name: &str,
    field_name: &str,
) -> Result<()> {
    if let Some(incoming) = incoming {
        match existing {
            Some(current) if current != &incoming => {
                return Err(Error::Stream(format!(
                    "conflicting {field_name} for {block_name} block at index {index}"
                )));
            }
            Some(_) => {}
            None => *existing = Some(incoming),
        }
    }
    Ok(())
}

#[inline]
fn append_or_replace(buffer: &mut String, chunk: String) {
    if buffer.is_empty() {
        *buffer = chunk;
    } else {
        buffer.push_str(&chunk);
    }
}

fn is_valid_json_document(input: &str) -> bool {
    let mut deserializer = serde_json::Deserializer::from_str(input);
    serde::de::IgnoredAny::deserialize(&mut deserializer).is_ok() && deserializer.end().is_ok()
}

fn parse_image_source_from_block_data(data: Option<ExtraMap>) -> Result<ImageSource> {
    let mut data = data
        .ok_or_else(|| Error::Stream("image block missing source during collection".to_string()))?;
    let source_value = data
        .remove("source")
        .ok_or_else(|| Error::Stream("image block missing source during collection".to_string()))?;
    serde_json::from_value(source_value).map_err(|e| {
        Error::Stream(format!(
            "image block source was invalid during collection: {e}"
        ))
    })
}

fn image_block_data(source: &ImageSource) -> ExtraMap {
    let mut data = ExtraMap::new();
    data.insert(
        "source".to_string(),
        serde_json::to_value(source).expect("ImageSource serialization should be infallible"),
    );
    data
}

impl ChatResponse {
    /// Convert a materialized response into the crate's standard normalized
    /// streaming adapter.
    ///
    /// This is primarily useful for provider and wrapper authors who implement
    /// one-shot `chat()` first and want `chat_stream()` to reuse the exact
    /// normalization logic from `anyllm` instead of reconstructing
    /// [`StreamEvent`] ordering by hand.
    ///
    /// Typed [`ResponseMetadata`](crate::ResponseMetadata) entries are flattened
    /// into portable JSON during this conversion. Reconstructing a
    /// [`ChatResponse`] from the emitted stream preserves the portable metadata
    /// map, but does not restore typed metadata entries.
    #[must_use]
    pub fn into_stream(self) -> SingleResponseStream {
        SingleResponseStream::new(self)
    }

    /// Convert a materialized response into the normalized event sequence used
    /// by [`SingleResponseStream`].
    ///
    /// This is useful when wrapper code needs to inspect, adapt, or re-emit the
    /// portable stream events directly before exposing them as a stream.
    ///
    /// Like [`ChatResponse::into_stream`], this exports response metadata through
    /// a portable JSON projection. Typed metadata that cannot be represented as
    /// portable JSON is skipped, and typed entries are not reconstructed if the
    /// events are collected back into a [`ChatResponse`].
    ///
    /// ```rust
    /// use anyllm::{ChatResponse, ContentBlock, FinishReason, StreamEvent};
    ///
    /// let events: Vec<_> = ChatResponse::new(vec![ContentBlock::Text {
    ///     text: "hello".to_string(),
    /// }])
    /// .finish_reason(FinishReason::Stop)
    /// .into_stream_events()
    /// .collect();
    ///
    /// assert!(matches!(events.first(), Some(StreamEvent::ResponseStart { .. })));
    /// assert!(matches!(events.last(), Some(StreamEvent::ResponseStop)));
    /// ```
    pub fn into_stream_events(self) -> impl Iterator<Item = StreamEvent> {
        self.stream_events()
    }

    pub(crate) fn stream_events(self) -> ResponseStreamEventIter {
        ResponseStreamEventIter::new(self)
    }
}

pub(crate) struct ResponseStreamEventIter {
    response_id: Option<String>,
    response_model: Option<String>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    metadata: ExtraMap,
    blocks: std::vec::IntoIter<ContentBlock>,
    next_index: usize,
    current_block: Option<PendingBlock>,
    state: ResponseStreamEventState,
}

impl ResponseStreamEventIter {
    fn new(response: ChatResponse) -> Self {
        let ChatResponse {
            content,
            finish_reason,
            usage,
            model,
            id,
            metadata,
        } = response;

        Self {
            response_id: id,
            response_model: model,
            finish_reason,
            usage,
            metadata: metadata.to_portable_map(),
            blocks: content.into_iter(),
            next_index: 0,
            current_block: None,
            state: ResponseStreamEventState::ResponseStart,
        }
    }

    fn advance_to_next_block(&mut self) {
        self.current_block = self.blocks.next().map(|block| {
            let pending = PendingBlock::new(self.next_index, block);
            self.next_index += 1;
            pending
        });

        self.state = if self.current_block.is_some() {
            ResponseStreamEventState::BlockStart
        } else {
            ResponseStreamEventState::ResponseMetadata
        };
    }
}

impl Iterator for ResponseStreamEventIter {
    type Item = StreamEvent;

    fn next(&mut self) -> Option<Self::Item> {
        match self.state {
            ResponseStreamEventState::ResponseStart => {
                self.advance_to_next_block();
                Some(StreamEvent::ResponseStart {
                    id: self.response_id.clone(),
                    model: self.response_model.clone(),
                })
            }
            ResponseStreamEventState::BlockStart => {
                let event = self.current_block.as_mut()?.start_event();
                self.state = if self.current_block.as_ref()?.has_delta() {
                    ResponseStreamEventState::BlockDelta
                } else {
                    ResponseStreamEventState::BlockStop
                };
                Some(event)
            }
            ResponseStreamEventState::BlockDelta => {
                let event = self.current_block.as_mut()?.delta_event();
                self.state = ResponseStreamEventState::BlockStop;
                Some(event)
            }
            ResponseStreamEventState::BlockStop => {
                let index = self.current_block.take()?.index();
                self.advance_to_next_block();
                Some(StreamEvent::BlockStop { index })
            }
            ResponseStreamEventState::ResponseMetadata => {
                self.state = ResponseStreamEventState::ResponseStop;
                Some(StreamEvent::ResponseMetadata {
                    finish_reason: self.finish_reason.take(),
                    usage: self.usage.take(),
                    usage_mode: UsageMetadataMode::Snapshot,
                    id: self.response_id.take(),
                    model: self.response_model.take(),
                    metadata: std::mem::take(&mut self.metadata),
                })
            }
            ResponseStreamEventState::ResponseStop => {
                self.state = ResponseStreamEventState::Done;
                Some(StreamEvent::ResponseStop)
            }
            ResponseStreamEventState::Done => None,
        }
    }
}

#[derive(Clone, Copy)]
enum ResponseStreamEventState {
    ResponseStart,
    BlockStart,
    BlockDelta,
    BlockStop,
    ResponseMetadata,
    ResponseStop,
    Done,
}

enum PendingBlock {
    Text {
        index: usize,
        text: String,
    },
    Image {
        index: usize,
        source: ImageSource,
    },
    ToolCall {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
    Reasoning {
        index: usize,
        text: String,
        signature: Option<String>,
    },
    Other {
        index: usize,
        type_name: Option<String>,
        data: Option<ExtraMap>,
    },
}

impl PendingBlock {
    fn new(index: usize, block: ContentBlock) -> Self {
        match block {
            ContentBlock::Text { text } => Self::Text { index, text },
            ContentBlock::Image { source } => Self::Image { index, source },
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => Self::ToolCall {
                index,
                id: Some(id),
                name: Some(name),
                arguments,
            },
            ContentBlock::Reasoning { text, signature } => Self::Reasoning {
                index,
                text,
                signature,
            },
            ContentBlock::Other { type_name, data } => Self::Other {
                index,
                type_name: Some(type_name),
                data: Some(data),
            },
        }
    }

    fn index(&self) -> usize {
        match self {
            Self::Text { index, .. }
            | Self::Image { index, .. }
            | Self::ToolCall { index, .. }
            | Self::Reasoning { index, .. }
            | Self::Other { index, .. } => *index,
        }
    }

    fn has_delta(&self) -> bool {
        !matches!(self, Self::Image { .. } | Self::Other { .. })
    }

    fn start_event(&mut self) -> StreamEvent {
        match self {
            Self::Text { index, .. } => StreamEvent::BlockStart {
                index: *index,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            },
            Self::Image { index, source } => StreamEvent::BlockStart {
                index: *index,
                block_type: StreamBlockType::Image,
                id: None,
                name: None,
                type_name: None,
                data: Some(image_block_data(source)),
            },
            Self::ToolCall {
                index, id, name, ..
            } => StreamEvent::BlockStart {
                index: *index,
                block_type: StreamBlockType::ToolCall,
                id: id.take(),
                name: name.take(),
                type_name: None,
                data: None,
            },
            Self::Reasoning { index, .. } => StreamEvent::BlockStart {
                index: *index,
                block_type: StreamBlockType::Reasoning,
                id: None,
                name: None,
                type_name: None,
                data: None,
            },
            Self::Other {
                index,
                type_name,
                data,
            } => StreamEvent::BlockStart {
                index: *index,
                block_type: StreamBlockType::Other,
                id: None,
                name: None,
                type_name: type_name.take(),
                data: data.take(),
            },
        }
    }

    fn delta_event(&mut self) -> StreamEvent {
        match self {
            Self::Text { index, text } => StreamEvent::TextDelta {
                index: *index,
                text: std::mem::take(text),
            },
            Self::ToolCall {
                index, arguments, ..
            } => StreamEvent::ToolCallDelta {
                index: *index,
                arguments: std::mem::take(arguments),
            },
            Self::Reasoning {
                index,
                text,
                signature,
            } => StreamEvent::ReasoningDelta {
                index: *index,
                text: std::mem::take(text),
                signature: signature.take(),
            },
            Self::Image { .. } => unreachable!("image blocks do not emit deltas"),
            Self::Other { .. } => unreachable!("other blocks do not emit deltas"),
        }
    }
}

/// Extension trait for consuming a [`ChatStream`] into a [`ChatResponse`].
pub trait ChatStreamExt: Stream<Item = Result<StreamEvent>> + Send {
    /// Collects all stream events into a complete [`ChatResponse`].
    ///
    /// # Errors
    ///
    /// Propagates stream item errors unchanged, and returns [`Error::Stream`]
    /// when the normalized event sequence is structurally incomplete (see
    /// [`StreamCollector::finish`]).
    fn collect_response(self) -> impl std::future::Future<Output = Result<ChatResponse>> + Send;

    /// Collects stream events into a partial response.
    ///
    /// Terminal stream errors are preserved on
    /// [`CollectedResponse::terminal_error`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Stream`] only if a normalized stream event violates
    /// collector invariants (for example, mismatched block types).
    fn collect_partial(self)
    -> impl std::future::Future<Output = Result<CollectedResponse>> + Send;
}

impl<S: Stream<Item = Result<StreamEvent>> + Send> ChatStreamExt for S {
    async fn collect_response(self) -> Result<ChatResponse> {
        use futures_util::StreamExt;
        let mut collector = StreamCollector::new();
        let mut stream = std::pin::pin!(self);
        while let Some(event) = stream.next().await {
            collector.push(event?)?;
        }
        collector.finish()
    }

    async fn collect_partial(self) -> Result<CollectedResponse> {
        use futures_util::StreamExt;
        let mut collector = StreamCollector::new();
        let mut stream = std::pin::pin!(self);
        while let Some(event) = stream.next().await {
            match event {
                Ok(event) => collector.push(event)?,
                Err(err) => {
                    collector.set_terminal_error(err);
                    break;
                }
            }
        }
        collector.finish_partial()
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;

    use futures_util::StreamExt;
    use serde_json::json;

    use super::*;

    #[derive(Clone, serde::Serialize)]
    struct DemoMetadata {
        request_id: String,
    }

    impl crate::ResponseMetadataType for DemoMetadata {
        const KEY: &'static str = "demo";
    }

    #[test]
    fn response_stream_event_iter_emits_expected_order() {
        let response = ChatResponse {
            content: vec![
                ContentBlock::Text {
                    text: "Hello".into(),
                },
                ContentBlock::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/cat.png".into(),
                    },
                },
                ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "search".into(),
                    arguments: r#"{"q":"rust"}"#.into(),
                },
                ContentBlock::Reasoning {
                    text: "Thinking".into(),
                    signature: Some("sig_1".into()),
                },
                ContentBlock::Other {
                    type_name: "citation".into(),
                    data: serde_json::Map::from_iter([(
                        "url".into(),
                        json!("https://example.com"),
                    )]),
                },
            ],
            finish_reason: Some(FinishReason::Stop),
            usage: Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(4),
                total_tokens: Some(14),
                ..Default::default()
            }),
            model: Some("mock".into()),
            id: Some("resp_1".into()),
            metadata: ResponseMetadata::new(),
        };

        let events: Vec<_> = response.stream_events().collect();

        assert_eq!(
            events,
            vec![
                StreamEvent::ResponseStart {
                    id: Some("resp_1".into()),
                    model: Some("mock".into()),
                },
                StreamEvent::BlockStart {
                    index: 0,
                    block_type: StreamBlockType::Text,
                    id: None,
                    name: None,
                    type_name: None,
                    data: None,
                },
                StreamEvent::TextDelta {
                    index: 0,
                    text: "Hello".into(),
                },
                StreamEvent::BlockStop { index: 0 },
                StreamEvent::BlockStart {
                    index: 1,
                    block_type: StreamBlockType::Image,
                    id: None,
                    name: None,
                    type_name: None,
                    data: Some(serde_json::Map::from_iter([(
                        "source".into(),
                        json!({
                            "type": "url",
                            "url": "https://example.com/cat.png"
                        }),
                    )])),
                },
                StreamEvent::BlockStop { index: 1 },
                StreamEvent::BlockStart {
                    index: 2,
                    block_type: StreamBlockType::ToolCall,
                    id: Some("call_1".into()),
                    name: Some("search".into()),
                    type_name: None,
                    data: None,
                },
                StreamEvent::ToolCallDelta {
                    index: 2,
                    arguments: r#"{"q":"rust"}"#.into(),
                },
                StreamEvent::BlockStop { index: 2 },
                StreamEvent::BlockStart {
                    index: 3,
                    block_type: StreamBlockType::Reasoning,
                    id: None,
                    name: None,
                    type_name: None,
                    data: None,
                },
                StreamEvent::ReasoningDelta {
                    index: 3,
                    text: "Thinking".into(),
                    signature: Some("sig_1".into()),
                },
                StreamEvent::BlockStop { index: 3 },
                StreamEvent::BlockStart {
                    index: 4,
                    block_type: StreamBlockType::Other,
                    id: None,
                    name: None,
                    type_name: Some("citation".into()),
                    data: Some(serde_json::Map::from_iter([(
                        "url".into(),
                        json!("https://example.com"),
                    )])),
                },
                StreamEvent::BlockStop { index: 4 },
                StreamEvent::ResponseMetadata {
                    finish_reason: Some(FinishReason::Stop),
                    usage: Some(Usage {
                        input_tokens: Some(10),
                        output_tokens: Some(4),
                        total_tokens: Some(14),
                        ..Default::default()
                    }),
                    usage_mode: UsageMetadataMode::Snapshot,
                    id: Some("resp_1".into()),
                    model: Some("mock".into()),
                    metadata: ExtraMap::new(),
                },
                StreamEvent::ResponseStop,
            ]
        );
    }

    #[test]
    fn collector_text_accumulation() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: None,
                model: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::TextDelta {
                index: 0,
                text: "Hello ".into(),
            })
            .unwrap();
        collector
            .push(StreamEvent::TextDelta {
                index: 0,
                text: "world!".into(),
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        collector
            .push(StreamEvent::ResponseMetadata {
                finish_reason: Some(FinishReason::Stop),
                usage: None,
                usage_mode: UsageMetadataMode::Snapshot,
                id: None,
                model: None,
                metadata: ExtraMap::new(),
            })
            .unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        assert_eq!(response.text(), Some("Hello world!".into()));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn collector_reconstructs_image_block_from_start_data() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("mock".into()),
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Image,
                id: None,
                name: None,
                type_name: None,
                data: Some(serde_json::Map::from_iter([(
                    "source".into(),
                    json!({
                        "type": "url",
                        "url": "https://example.com/cat.png"
                    }),
                )])),
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        assert!(matches!(
            &response.content[..],
            [ContentBlock::Image {
                source: ImageSource::Url { url }
            }] if url == "https://example.com/cat.png"
        ));
    }

    #[test]
    fn collector_late_block_start_preserves_prior_text_delta() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: None,
                model: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::TextDelta {
                index: 0,
                text: "Hello".into(),
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        assert_eq!(response.text(), Some("Hello".into()));
    }

    #[test]
    fn collector_late_tool_call_start_merges_missing_identity() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: None,
                model: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::ToolCallDelta {
                index: 0,
                arguments: r#"{"q":"rust"}"#.into(),
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::ToolCall,
                id: Some("call_1".into()),
                name: Some("search".into()),
                type_name: None,
                data: None,
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        let calls: Vec<_> = response.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, r#"{"q":"rust"}"#);
    }

    #[test]
    fn collector_rejects_duplicate_block_start_for_same_index() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            })
            .unwrap();

        let err = collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            })
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Stream(message) if message == "duplicate block_start for text block at index 0"
        ));
    }

    #[test]
    fn collector_rejects_duplicate_block_stop_for_same_index() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 0 }).unwrap();

        let err = collector
            .push(StreamEvent::BlockStop { index: 0 })
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Stream(message) if message == "duplicate block_stop for text block at index 0"
        ));
    }

    #[test]
    fn collector_rejects_duplicate_response_stop() {
        let mut collector = StreamCollector::new();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let err = collector.push(StreamEvent::ResponseStop).unwrap_err();
        assert!(matches!(
            err,
            Error::Stream(message) if message == "duplicate response_stop"
        ));
    }

    #[test]
    fn collector_rejects_events_after_response_stop() {
        let mut collector = StreamCollector::new();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let err = collector
            .push(StreamEvent::TextDelta {
                index: 0,
                text: "late".into(),
            })
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Stream(message) if message == "text_delta received after response_stop"
        ));
    }

    #[test]
    fn collector_rejects_conflicting_tool_call_start_metadata() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ToolCallDelta {
                index: 0,
                arguments: "{}".into(),
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::ToolCall,
                id: Some("call_1".into()),
                name: Some("search".into()),
                type_name: None,
                data: None,
            })
            .unwrap();

        let err = collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::ToolCall,
                id: Some("call_2".into()),
                name: Some("search".into()),
                type_name: None,
                data: None,
            })
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Stream(message)
                if message == "duplicate block_start for tool_call block at index 0"
        ));
    }

    #[test]
    fn collector_push_ref_matches_owned_push_for_late_tool_call_start() {
        let events = vec![
            StreamEvent::ResponseStart {
                id: Some("resp_ref".into()),
                model: Some("mock".into()),
            },
            StreamEvent::ToolCallDelta {
                index: 0,
                arguments: r#"{"q":"rust"}"#.into(),
            },
            StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::ToolCall,
                id: Some("call_1".into()),
                name: Some("search".into()),
                type_name: None,
                data: None,
            },
            StreamEvent::ResponseMetadata {
                finish_reason: Some(FinishReason::ToolCalls),
                usage: Some(Usage {
                    input_tokens: Some(2),
                    output_tokens: Some(1),
                    ..Default::default()
                }),
                usage_mode: UsageMetadataMode::Snapshot,
                id: Some("resp_final".into()),
                model: Some("mock-final".into()),
                metadata: ExtraMap::from_iter([("trace".into(), json!(true))]),
            },
            StreamEvent::BlockStop { index: 0 },
            StreamEvent::ResponseStop,
        ];

        let mut owned = StreamCollector::new();
        for event in events.clone() {
            owned.push(event).unwrap();
        }

        let mut by_ref = StreamCollector::new();
        for event in &events {
            by_ref.push_ref(event).unwrap();
        }

        let owned = owned.finish().unwrap();
        let by_ref = by_ref.finish().unwrap();
        assert_eq!(owned.content, by_ref.content);
        assert_eq!(owned.id, by_ref.id);
        assert_eq!(owned.model, by_ref.model);
        assert_eq!(owned.finish_reason, by_ref.finish_reason);
        assert_eq!(owned.usage, by_ref.usage);
        assert_eq!(
            owned.metadata.get_portable("trace"),
            by_ref.metadata.get_portable("trace")
        );
    }

    #[test]
    fn collector_tool_call_reassembly_with_interleaving() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: None,
                model: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::ToolCall,
                id: Some("call_1".into()),
                name: Some("search".into()),
                type_name: None,
                data: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::ToolCallDelta {
                index: 0,
                arguments: r#"{"q":"#.into(),
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 1,
                block_type: StreamBlockType::ToolCall,
                id: Some("call_2".into()),
                name: Some("read".into()),
                type_name: None,
                data: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::ToolCallDelta {
                index: 1,
                arguments: r#"{"path":"foo"}"#.into(),
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 1 }).unwrap();
        collector
            .push(StreamEvent::ToolCallDelta {
                index: 0,
                arguments: "\"rust\"}".into(),
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        let calls: Vec<_> = response.tool_calls().collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, r#"{"q":"rust"}"#);
        assert_eq!(calls[1].arguments, r#"{"path":"foo"}"#);
    }

    #[test]
    fn collector_preserves_metadata() {
        let mut collector = StreamCollector::new();
        let mut metadata = ExtraMap::new();
        metadata.insert("provider_request_id".into(), json!("req_123"));
        collector
            .push(StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("gpt-4o".into()),
            })
            .unwrap();
        collector
            .push(StreamEvent::ResponseMetadata {
                finish_reason: Some(FinishReason::Stop),
                usage: Some(Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(5),
                    ..Default::default()
                }),
                usage_mode: UsageMetadataMode::Snapshot,
                id: None,
                model: None,
                metadata,
            })
            .unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        assert_eq!(response.id.as_deref(), Some("resp_1"));
        assert_eq!(response.model.as_deref(), Some("gpt-4o"));
        assert_eq!(response.usage.unwrap().total(), Some(15));
        assert_eq!(
            response.metadata.get_portable("provider_request_id"),
            Some(&json!("req_123"))
        );
    }

    #[test]
    fn collector_replaces_usage_for_snapshot_metadata() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: Some("resp_usage".into()),
                model: Some("mock".into()),
            })
            .unwrap();
        collector
            .push(StreamEvent::ResponseMetadata {
                finish_reason: None,
                usage: Some(Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(3),
                    ..Default::default()
                }),
                usage_mode: UsageMetadataMode::Snapshot,
                id: None,
                model: None,
                metadata: ExtraMap::new(),
            })
            .unwrap();
        collector
            .push(StreamEvent::ResponseMetadata {
                finish_reason: Some(FinishReason::Stop),
                usage: Some(Usage {
                    input_tokens: Some(12),
                    output_tokens: Some(7),
                    ..Default::default()
                }),
                usage_mode: UsageMetadataMode::Snapshot,
                id: None,
                model: None,
                metadata: ExtraMap::new(),
            })
            .unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.output_tokens, Some(7));
    }

    #[test]
    fn collector_accumulates_usage_for_delta_metadata() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: Some("resp_delta".into()),
                model: Some("mock".into()),
            })
            .unwrap();
        collector
            .push(StreamEvent::ResponseMetadata {
                finish_reason: None,
                usage: Some(Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(3),
                    ..Default::default()
                }),
                usage_mode: UsageMetadataMode::Delta,
                id: None,
                model: None,
                metadata: ExtraMap::new(),
            })
            .unwrap();
        collector
            .push(StreamEvent::ResponseMetadata {
                finish_reason: Some(FinishReason::Stop),
                usage: Some(Usage {
                    input_tokens: Some(2),
                    output_tokens: Some(4),
                    ..Default::default()
                }),
                usage_mode: UsageMetadataMode::Delta,
                id: None,
                model: None,
                metadata: ExtraMap::new(),
            })
            .unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.output_tokens, Some(7));
    }

    #[test]
    fn collector_accumulates_reasoning_and_signature() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: Some("resp_reasoning".into()),
                model: Some("mock".into()),
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Reasoning,
                id: None,
                name: None,
                type_name: None,
                data: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::ReasoningDelta {
                index: 0,
                text: "Let me ".into(),
                signature: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::ReasoningDelta {
                index: 0,
                text: "think".into(),
                signature: Some("sig_123".into()),
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        assert_eq!(response.reasoning_text(), Some("Let me think".into()));
        assert!(matches!(
            response.content.as_slice(),
            [ContentBlock::Reasoning { text, signature }]
                if text == "Let me think" && signature.as_deref() == Some("sig_123")
        ));
    }

    #[test]
    fn collector_merges_metadata_chunks_and_latest_identity_wins() {
        let mut collector = StreamCollector::new();
        let mut first_metadata = ExtraMap::new();
        first_metadata.insert("provider_region".into(), json!("us-east-1"));
        let mut second_metadata = ExtraMap::new();
        second_metadata.insert("provider_trace".into(), json!("trace_123"));
        collector
            .push(StreamEvent::ResponseStart {
                id: Some("resp_initial".into()),
                model: Some("model-initial".into()),
            })
            .unwrap();
        collector
            .push(StreamEvent::ResponseMetadata {
                finish_reason: None,
                usage: Some(Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(3),
                    ..Default::default()
                }),
                usage_mode: UsageMetadataMode::Delta,
                id: None,
                model: None,
                metadata: first_metadata,
            })
            .unwrap();
        collector
            .push(StreamEvent::ResponseMetadata {
                finish_reason: Some(FinishReason::Stop),
                usage: Some(Usage {
                    input_tokens: Some(2),
                    output_tokens: Some(4),
                    ..Default::default()
                }),
                usage_mode: UsageMetadataMode::Delta,
                id: Some("resp_final".into()),
                model: Some("model-final".into()),
                metadata: second_metadata,
            })
            .unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.id.as_deref(), Some("resp_final"));
        assert_eq!(response.model.as_deref(), Some("model-final"));

        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.output_tokens, Some(7));
        assert_eq!(
            response.metadata.get_portable("provider_region"),
            Some(&json!("us-east-1"))
        );
        assert_eq!(
            response.metadata.get_portable("provider_trace"),
            Some(&json!("trace_123"))
        );
    }

    #[test]
    fn response_stream_event_iter_preserves_portable_metadata() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert_portable("provider_request_id", json!("req_789"));

        let response = ChatResponse {
            content: vec![ContentBlock::Text { text: "ok".into() }],
            finish_reason: Some(FinishReason::Stop),
            usage: None,
            model: Some("mock".into()),
            id: Some("resp_meta".into()),
            metadata,
        };

        let events: Vec<_> = response.stream_events().collect();
        assert!(matches!(
            &events[4],
            StreamEvent::ResponseMetadata { metadata, .. }
                if metadata.get("provider_request_id") == Some(&json!("req_789"))
        ));
    }

    #[test]
    fn chat_response_into_stream_events_emits_same_normalized_sequence() {
        let response = ChatResponse {
            content: vec![
                ContentBlock::Text {
                    text: "Hello".into(),
                },
                ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "search".into(),
                    arguments: r#"{"q":"rust"}"#.into(),
                },
            ],
            finish_reason: Some(FinishReason::ToolCalls),
            usage: Some(Usage::new().input_tokens(10).output_tokens(3)),
            model: Some("mock".into()),
            id: Some("resp_events".into()),
            metadata: ResponseMetadata::new(),
        };

        let events: Vec<_> = response.into_stream_events().collect();

        assert!(matches!(
            events.as_slice(),
            [
                StreamEvent::ResponseStart { .. },
                StreamEvent::BlockStart { index: 0, .. },
                StreamEvent::TextDelta { index: 0, .. },
                StreamEvent::BlockStop { index: 0 },
                StreamEvent::BlockStart { index: 1, .. },
                StreamEvent::ToolCallDelta { index: 1, .. },
                StreamEvent::BlockStop { index: 1 },
                StreamEvent::ResponseMetadata { .. },
                StreamEvent::ResponseStop,
            ]
        ));
    }

    #[tokio::test]
    async fn chat_response_into_stream_round_trips_full_response() {
        let response = ChatResponse {
            content: vec![
                ContentBlock::Reasoning {
                    text: "Thinking".into(),
                    signature: Some("sig_1".into()),
                },
                ContentBlock::Text {
                    text: "Hello".into(),
                },
            ],
            finish_reason: Some(FinishReason::Stop),
            usage: Some(Usage::new().input_tokens(8).output_tokens(2)),
            model: Some("mock".into()),
            id: Some("resp_into_stream".into()),
            metadata: ResponseMetadata::new(),
        };

        let collected = response.into_stream().collect_response().await.unwrap();

        assert_eq!(collected.reasoning_text(), Some("Thinking".into()));
        assert_eq!(collected.text(), Some("Hello".into()));
        assert_eq!(collected.finish_reason, Some(FinishReason::Stop));
        assert_eq!(collected.id.as_deref(), Some("resp_into_stream"));
    }

    #[tokio::test]
    async fn chat_response_into_stream_preserves_portable_metadata_but_drops_typed_entries() {
        let mut metadata = ResponseMetadata::new();
        metadata.insert(DemoMetadata {
            request_id: "req_123".into(),
        });
        metadata.insert_portable("provider_request_id", json!("req_portable"));

        let response = ChatResponse {
            content: vec![ContentBlock::Text {
                text: "Hello".into(),
            }],
            finish_reason: Some(FinishReason::Stop),
            usage: None,
            model: Some("mock".into()),
            id: Some("resp_metadata_roundtrip".into()),
            metadata,
        };

        let collected = response.into_stream().collect_response().await.unwrap();

        assert!(collected.metadata.get::<DemoMetadata>().is_none());
        assert_eq!(
            collected.metadata.get_portable("provider_request_id"),
            Some(&json!("req_portable"))
        );
        assert_eq!(
            collected.metadata.get_portable("demo"),
            Some(&json!({"request_id": "req_123"}))
        );
    }

    #[tokio::test]
    async fn single_response_stream_round_trips_full_response() {
        let response = ChatResponse {
            content: vec![
                ContentBlock::Reasoning {
                    text: "Thinking".into(),
                    signature: Some("sig_1".into()),
                },
                ContentBlock::Text {
                    text: "Hello".into(),
                },
                ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "search".into(),
                    arguments: r#"{"q":"rust"}"#.into(),
                },
                ContentBlock::Other {
                    type_name: "citation".into(),
                    data: ExtraMap::new(),
                },
            ],
            finish_reason: Some(FinishReason::ToolCalls),
            usage: Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(5),
                ..Default::default()
            }),
            model: Some("mock".into()),
            id: Some("resp_123".into()),
            metadata: ResponseMetadata::new(),
        };

        let collected = SingleResponseStream::new(response)
            .collect_response()
            .await
            .unwrap();

        assert_eq!(collected.reasoning_text(), Some("Thinking".into()));
        assert_eq!(collected.text(), Some("Hello".into()));
        assert_eq!(collected.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(collected.id.as_deref(), Some("resp_123"));
        assert_eq!(collected.model.as_deref(), Some("mock"));
        let calls: Vec<_> = collected.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, r#"{"q":"rust"}"#);
        assert!(matches!(
            collected.content.as_slice(),
            [
                ContentBlock::Reasoning { .. },
                ContentBlock::Text { .. },
                ContentBlock::ToolCall { .. },
                ContentBlock::Other { type_name, data }
            ] if type_name == "citation" && data.is_empty()
        ));
    }

    #[tokio::test]
    async fn single_response_stream_yields_only_ok_events() {
        let response = ChatResponse {
            content: vec![ContentBlock::Text { text: "ok".into() }],
            finish_reason: Some(FinishReason::Stop),
            usage: None,
            model: None,
            id: None,
            metadata: ResponseMetadata::new(),
        };

        let mut stream = pin!(SingleResponseStream::from(response));
        while let Some(event) = stream.next().await {
            assert!(event.is_ok());
        }
    }

    #[test]
    fn collector_other_block_defaults_type_name_and_data() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: None,
                model: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Other,
                id: None,
                name: None,
                type_name: None,
                data: None,
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        assert!(matches!(
            response.content.as_slice(),
            [ContentBlock::Other { type_name, data }] if type_name == "other" && data.is_empty()
        ));
    }

    #[test]
    fn collector_other_block_preserves_explicit_type_and_data() {
        let mut collector = StreamCollector::new();
        let mut data = ExtraMap::new();
        data.insert("url".into(), json!("https://example.com"));

        collector
            .push(StreamEvent::ResponseStart {
                id: None,
                model: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Other,
                id: None,
                name: None,
                type_name: Some("citation".into()),
                data: Some(data),
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        assert!(matches!(
            response.content.as_slice(),
            [ContentBlock::Other { type_name, data }]
                if type_name == "citation"
                    && data.get("url") == Some(&json!("https://example.com"))
        ));
    }

    #[test]
    fn collector_errors_on_mismatched_delta_type() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            })
            .unwrap();

        let err = collector
            .push(StreamEvent::ToolCallDelta {
                index: 0,
                arguments: "{}".into(),
            })
            .unwrap_err();
        assert!(matches!(err, Error::Stream(_)));
    }

    #[test]
    fn collector_finish_requires_explicit_response_stop() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("mock".into()),
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::TextDelta {
                index: 0,
                text: "partial".into(),
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        collector
            .push(StreamEvent::ResponseMetadata {
                finish_reason: Some(FinishReason::Stop),
                usage: None,
                usage_mode: UsageMetadataMode::Snapshot,
                id: None,
                model: None,
                metadata: ExtraMap::new(),
            })
            .unwrap();

        let err = collector.finish().unwrap_err();
        assert!(matches!(err, Error::Stream(_)));
    }

    #[test]
    fn collector_finish_errors_on_open_text_block() {
        let events = vec![
            StreamEvent::ResponseStart {
                id: Some("resp_text".into()),
                model: Some("mock".into()),
            },
            StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            },
            StreamEvent::TextDelta {
                index: 0,
                text: "partial text".into(),
            },
            StreamEvent::ResponseStop,
        ];

        let mut collector = StreamCollector::new();
        for event in events.iter().cloned() {
            collector.push(event).unwrap();
        }

        let err = collector.finish().unwrap_err();
        assert!(matches!(
            err,
            Error::Stream(message)
                if message
                    == "stream incomplete: start=true, stop=true, open_blocks=true, dropped_incomplete_tool_calls=false, terminal_error=none"
        ));

        let mut collector = StreamCollector::new();
        for event in events {
            collector.push(event).unwrap();
        }
        let collected = collector.finish_partial().unwrap();
        assert_eq!(collected.response.text(), Some("partial text".into()));
        assert!(collected.completeness.has_open_blocks);
        assert!(!collected.completeness.is_complete());
        assert!(collected.terminal_error.is_none());
    }

    #[test]
    fn collector_finish_errors_on_open_reasoning_block() {
        let events = vec![
            StreamEvent::ResponseStart {
                id: Some("resp_reasoning".into()),
                model: Some("mock".into()),
            },
            StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Reasoning,
                id: None,
                name: None,
                type_name: None,
                data: None,
            },
            StreamEvent::ReasoningDelta {
                index: 0,
                text: "thinking".into(),
                signature: Some("sig_open".into()),
            },
            StreamEvent::ResponseStop,
        ];

        let mut collector = StreamCollector::new();
        for event in events.iter().cloned() {
            collector.push(event).unwrap();
        }

        let err = collector.finish().unwrap_err();
        assert!(matches!(
            err,
            Error::Stream(message)
                if message
                    == "stream incomplete: start=true, stop=true, open_blocks=true, dropped_incomplete_tool_calls=false, terminal_error=none"
        ));

        let mut collector = StreamCollector::new();
        for event in events {
            collector.push(event).unwrap();
        }
        let collected = collector.finish_partial().unwrap();
        assert!(matches!(
            collected.response.content.as_slice(),
            [ContentBlock::Reasoning { text, signature }]
                if text == "thinking" && signature.as_deref() == Some("sig_open")
        ));
        assert!(collected.completeness.has_open_blocks);
        assert!(!collected.completeness.is_complete());
        assert!(collected.terminal_error.is_none());
    }

    #[test]
    fn collector_finish_errors_on_open_other_block() {
        let events = vec![
            StreamEvent::ResponseStart {
                id: Some("resp_other".into()),
                model: Some("mock".into()),
            },
            StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Other,
                id: None,
                name: None,
                type_name: Some("citation".into()),
                data: Some(ExtraMap::from_iter([(
                    "url".into(),
                    json!("https://example.com/open"),
                )])),
            },
            StreamEvent::ResponseStop,
        ];

        let mut collector = StreamCollector::new();
        for event in events.iter().cloned() {
            collector.push(event).unwrap();
        }

        let err = collector.finish().unwrap_err();
        assert!(matches!(
            err,
            Error::Stream(message)
                if message
                    == "stream incomplete: start=true, stop=true, open_blocks=true, dropped_incomplete_tool_calls=false, terminal_error=none"
        ));

        let mut collector = StreamCollector::new();
        for event in events {
            collector.push(event).unwrap();
        }
        let collected = collector.finish_partial().unwrap();
        assert!(matches!(
            collected.response.content.as_slice(),
            [ContentBlock::Other { type_name, data }]
                if type_name == "citation"
                    && data.get("url") == Some(&json!("https://example.com/open"))
        ));
        assert!(collected.completeness.has_open_blocks);
        assert!(!collected.completeness.is_complete());
        assert!(collected.terminal_error.is_none());
    }

    #[test]
    fn collector_finish_errors_on_tool_call_missing_name() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("mock".into()),
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::ToolCall,
                id: Some("call_1".into()),
                name: None,
                type_name: None,
                data: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::ToolCallDelta {
                index: 0,
                arguments: "{}".into(),
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let err = collector.finish().unwrap_err();
        assert!(matches!(
            err,
            Error::Stream(message)
                if message == "tool call block missing id or name during collection"
        ));
    }

    #[test]
    fn collector_finish_errors_on_open_tool_call_block() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("mock".into()),
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::ToolCall,
                id: Some("call_1".into()),
                name: Some("search".into()),
                type_name: None,
                data: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::ToolCallDelta {
                index: 0,
                arguments: "{}".into(),
            })
            .unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let err = collector.finish().unwrap_err();
        assert!(matches!(
            err,
            Error::Stream(message)
                if message == "tool call block remained open during collection"
        ));
    }

    #[test]
    fn collector_defaults_empty_tool_call_arguments_to_empty_object() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("mock".into()),
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::ToolCall,
                id: Some("call_1".into()),
                name: Some("get_time".into()),
                type_name: None,
                data: None,
            })
            .unwrap();
        collector.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        collector.push(StreamEvent::ResponseStop).unwrap();

        let response = collector.finish().unwrap();
        let calls: Vec<_> = response.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn collector_finish_partial_reports_incomplete_stream() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: Some("resp_partial".into()),
                model: Some("mock".into()),
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::TextDelta {
                index: 0,
                text: "partial".into(),
            })
            .unwrap();

        let collected = collector.finish_partial().unwrap();
        assert_eq!(collected.response.text(), Some("partial".into()));
        assert!(!collected.completeness.saw_response_stop);
        assert!(collected.completeness.has_open_blocks);
        assert!(!collected.completeness.is_complete());
        assert!(collected.terminal_error.is_none());
    }

    #[test]
    fn collector_finish_partial_drops_incomplete_tool_call() {
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("mock".into()),
            })
            .unwrap();
        collector
            .push(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::ToolCall,
                id: Some("call_1".into()),
                name: Some("search".into()),
                type_name: None,
                data: None,
            })
            .unwrap();
        collector
            .push(StreamEvent::ToolCallDelta {
                index: 0,
                arguments: r#"{"q":"rust"}"#.into(),
            })
            .unwrap();

        let collected = collector.finish_partial().unwrap();
        assert!(!collected.response.has_tool_calls());
        assert!(collected.completeness.dropped_incomplete_tool_calls);
        assert!(collected.completeness.has_open_blocks);
        assert!(collected.terminal_error.is_none());
    }

    #[tokio::test]
    async fn chat_stream_ext_collect_response() {
        let events: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("mock".into()),
            }),
            Ok(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            }),
            Ok(StreamEvent::TextDelta {
                index: 0,
                text: "Hello ".into(),
            }),
            Ok(StreamEvent::TextDelta {
                index: 0,
                text: "world".into(),
            }),
            Ok(StreamEvent::BlockStop { index: 0 }),
            Ok(StreamEvent::ResponseMetadata {
                finish_reason: Some(FinishReason::Stop),
                usage: Some(Usage {
                    input_tokens: Some(10),
                    output_tokens: Some(5),
                    ..Default::default()
                }),
                usage_mode: UsageMetadataMode::Snapshot,
                id: None,
                model: None,
                metadata: ExtraMap::new(),
            }),
            Ok(StreamEvent::ResponseStop),
        ];

        let response = futures::stream::iter(events)
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text(), Some("Hello world".into()));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.id.as_deref(), Some("resp_1"));
    }

    #[tokio::test]
    async fn chat_stream_ext_propagates_stream_errors() {
        let events: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("mock".into()),
            }),
            Err(Error::Timeout("slow stream".into())),
        ];

        match futures::stream::iter(events).collect_response().await {
            Err(Error::Timeout(message)) => assert_eq!(message, "slow stream"),
            Err(other) => panic!("expected timeout error, got {other:?}"),
            Ok(_) => panic!("expected collect_response to return an error"),
        }
    }

    #[tokio::test]
    async fn chat_stream_ext_collect_partial_recovers_terminal_stream_error() {
        let events: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("mock".into()),
            }),
            Ok(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            }),
            Ok(StreamEvent::TextDelta {
                index: 0,
                text: "Hello".into(),
            }),
            Err(Error::Timeout("slow stream".into())),
        ];

        let collected = futures::stream::iter(events)
            .collect_partial()
            .await
            .unwrap();
        assert_eq!(collected.response.text(), Some("Hello".into()));
        assert!(matches!(
            collected.terminal_error,
            Some(Error::Timeout(message)) if message == "slow stream"
        ));
        assert!(!collected.completeness.is_complete());
    }

    #[tokio::test]
    async fn chat_stream_ext_collect_partial_preserves_complete_streams() {
        let events: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::ResponseStart {
                id: Some("resp_1".into()),
                model: Some("mock".into()),
            }),
            Ok(StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            }),
            Ok(StreamEvent::TextDelta {
                index: 0,
                text: "ok".into(),
            }),
            Ok(StreamEvent::BlockStop { index: 0 }),
            Ok(StreamEvent::ResponseStop),
        ];

        let collected = futures::stream::iter(events)
            .collect_partial()
            .await
            .unwrap();
        assert_eq!(collected.response.text(), Some("ok".into()));
        assert!(collected.completeness.is_complete());
        assert!(collected.terminal_error.is_none());
    }
}
