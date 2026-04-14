use std::collections::{HashMap, VecDeque};

use anyllm::{ChatStream, ExtraMap, StreamBlockType, StreamEvent, UsageMetadataMode};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;

use crate::wire::{ChatCompletionChunk, from_api_usage, parse_finish_reason};

pub struct SseState {
    started_response: bool,
    response_id: Option<String>,
    model: Option<String>,
    tool_calls: HashMap<usize, PendingToolCall>,
    open_text_block: bool,
}

#[derive(Default)]
struct PendingToolCall {
    id: Option<String>,
    name: Option<String>,
    buffered_arguments: String,
    started: bool,
}

impl PendingToolCall {
    fn is_resolved(&self) -> bool {
        self.id.is_some() && self.name.is_some()
    }

    fn has_pending_content(&self) -> bool {
        self.id.is_some() || self.name.is_some() || !self.buffered_arguments.is_empty()
    }
}

impl Default for SseState {
    fn default() -> Self {
        Self::new()
    }
}

impl SseState {
    pub fn new() -> Self {
        Self {
            started_response: false,
            response_id: None,
            model: None,
            tool_calls: HashMap::new(),
            open_text_block: false,
        }
    }

    fn ensure_response_started(
        &mut self,
        chunk: &ChatCompletionChunk,
        events: &mut Vec<anyllm::Result<StreamEvent>>,
    ) {
        if self.started_response {
            if self.response_id.is_none() {
                self.response_id = Some(chunk.id.clone());
            }
            if self.model.is_none() {
                self.model = chunk.model.clone();
            }
            return;
        }

        self.started_response = true;
        self.response_id = Some(chunk.id.clone());
        self.model = chunk.model.clone();
        events.push(Ok(StreamEvent::ResponseStart {
            id: Some(chunk.id.clone()),
            model: chunk.model.clone(),
        }));
    }

    fn ensure_text_block_open(&mut self, events: &mut Vec<anyllm::Result<StreamEvent>>) {
        if self.open_text_block {
            return;
        }

        self.open_text_block = true;
        events.push(Ok(StreamEvent::BlockStart {
            index: 0,
            block_type: StreamBlockType::Text,
            id: None,
            name: None,
            type_name: None,
            data: None,
        }));
    }

    fn close_text_block(&mut self, events: &mut Vec<anyllm::Result<StreamEvent>>) {
        if !self.open_text_block {
            return;
        }

        self.open_text_block = false;
        events.push(Ok(StreamEvent::BlockStop { index: 0 }));
    }

    fn flush_tool_call_start(
        pending: &mut PendingToolCall,
        index: usize,
        events: &mut Vec<anyllm::Result<StreamEvent>>,
    ) {
        if pending.started || !pending.is_resolved() {
            return;
        }

        pending.started = true;
        events.push(Ok(StreamEvent::BlockStart {
            index,
            block_type: StreamBlockType::ToolCall,
            id: pending.id.clone(),
            name: pending.name.clone(),
            type_name: None,
            data: None,
        }));

        if !pending.buffered_arguments.is_empty() {
            events.push(Ok(StreamEvent::ToolCallDelta {
                index,
                arguments: std::mem::take(&mut pending.buffered_arguments),
            }));
        }
    }

    fn finalize_tool_call(
        index: usize,
        mut pending: PendingToolCall,
        events: &mut Vec<anyllm::Result<StreamEvent>>,
    ) {
        if !pending.started {
            if !pending.has_pending_content() {
                return;
            }

            events.push(Ok(StreamEvent::BlockStart {
                index,
                block_type: StreamBlockType::ToolCall,
                id: pending.id.take(),
                name: pending.name.take(),
                type_name: None,
                data: None,
            }));
            pending.started = true;
        }

        if !pending.buffered_arguments.is_empty() {
            events.push(Ok(StreamEvent::ToolCallDelta {
                index,
                arguments: pending.buffered_arguments,
            }));
        }

        events.push(Ok(StreamEvent::BlockStop { index }));
    }

    fn close_all_tool_calls(&mut self, events: &mut Vec<anyllm::Result<StreamEvent>>) {
        let mut indices: Vec<_> = self.tool_calls.keys().copied().collect();
        indices.sort_unstable();
        for index in indices {
            if let Some(pending) = self.tool_calls.remove(&index) {
                Self::finalize_tool_call(index, pending, events);
            }
        }
    }

    fn finalize_blocks(&mut self, events: &mut Vec<anyllm::Result<StreamEvent>>) {
        self.close_text_block(events);
        self.close_all_tool_calls(events);
    }

    fn finalize_complete(&mut self) -> Vec<anyllm::Result<StreamEvent>> {
        let mut events = Vec::new();
        self.finalize_blocks(&mut events);
        if self.started_response {
            events.push(Ok(StreamEvent::ResponseStop));
            self.started_response = false;
        }
        events
    }

    fn finalize_incomplete(&mut self) -> Vec<anyllm::Result<StreamEvent>> {
        let mut events = Vec::new();
        self.finalize_blocks(&mut events);
        self.started_response = false;
        events
    }

    fn merge_tool_field(
        current: &mut Option<String>,
        incoming: Option<String>,
        index: usize,
        field_name: &str,
        events: &mut Vec<anyllm::Result<StreamEvent>>,
    ) {
        if let Some(incoming) = incoming {
            match current {
                Some(existing) if existing != &incoming => {
                    events.push(Err(anyllm::Error::Stream(format!(
                        "conflicting {field_name} for tool call block at index {index}"
                    ))));
                }
                Some(_) => {}
                None => *current = Some(incoming),
            }
        }
    }
}

impl SseState {
    fn process_sse_data(&mut self, data: &str) -> Vec<anyllm::Result<StreamEvent>> {
        if data.is_empty() {
            return Vec::new();
        }

        if data == "[DONE]" {
            return self.finalize_complete();
        }

        let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                return vec![Err(anyllm::Error::Stream(format!("SSE parse error: {e}")))];
            }
        };

        let mut events: Vec<anyllm::Result<StreamEvent>> = Vec::new();
        self.ensure_response_started(&chunk, &mut events);

        for choice in &chunk.choices {
            let delta = &choice.delta;

            if let Some(ref text) = delta.content
                && !text.is_empty()
            {
                self.ensure_text_block_open(&mut events);
                events.push(Ok(StreamEvent::TextDelta {
                    index: 0,
                    text: text.clone(),
                }));
            }

            if let Some(ref tool_calls) = delta.tool_calls {
                self.close_text_block(&mut events);

                for tc in tool_calls {
                    let index = tc.index as usize + 1;

                    let pending = self.tool_calls.entry(index).or_default();
                    Self::merge_tool_field(
                        &mut pending.id,
                        tc.id.clone(),
                        index,
                        "id",
                        &mut events,
                    );

                    if let Some(ref func) = tc.function {
                        Self::merge_tool_field(
                            &mut pending.name,
                            func.name.clone(),
                            index,
                            "name",
                            &mut events,
                        );

                        if !pending.started {
                            Self::flush_tool_call_start(pending, index, &mut events);
                        }

                        if let Some(ref args) = func.arguments
                            && !args.is_empty()
                        {
                            if pending.started {
                                events.push(Ok(StreamEvent::ToolCallDelta {
                                    index,
                                    arguments: args.clone(),
                                }));
                            } else {
                                pending.buffered_arguments.push_str(args);
                            }
                        }
                    } else if !pending.started {
                        Self::flush_tool_call_start(pending, index, &mut events);
                    }
                }
            }

            if let Some(ref reason) = choice.finish_reason {
                self.close_text_block(&mut events);
                self.close_all_tool_calls(&mut events);
                // OpenAI-compatible chunk finish reasons are response-level state
                // transitions, not usage updates, so they carry no usage payload.
                events.push(Ok(StreamEvent::ResponseMetadata {
                    finish_reason: Some(parse_finish_reason(reason)),
                    usage: None,
                    usage_mode: UsageMetadataMode::Snapshot,
                    id: self.response_id.clone(),
                    model: self.model.clone(),
                    metadata: ExtraMap::new(),
                }));
            }
        }

        if let Some(ref usage) = chunk.usage {
            // OpenAI-compatible chunk usage is cumulative for the response and
            // should replace any previously seen usage snapshot.
            events.push(Ok(StreamEvent::ResponseMetadata {
                finish_reason: None,
                usage: Some(from_api_usage(usage)),
                usage_mode: UsageMetadataMode::Snapshot,
                id: self.response_id.clone(),
                model: self.model.clone(),
                metadata: ExtraMap::new(),
            }));
        }

        events
    }
}

pub fn process_sse_data(state: &mut SseState, data: &str) -> Vec<anyllm::Result<StreamEvent>> {
    state.process_sse_data(data)
}

pub fn sse_to_stream<S, B, E, F>(byte_stream: S, map_stream_error: F) -> ChatStream
where
    S: futures_core::Stream<Item = Result<B, E>> + Send + Unpin + 'static,
    B: AsRef<[u8]>,
    E: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
    F: Fn(eventsource_stream::EventStreamError<E>) -> anyllm::Error + Send + Sync + Copy + 'static,
{
    let event_stream = byte_stream.eventsource();

    Box::pin(futures_util::stream::unfold(
        (
            event_stream,
            SseState::new(),
            VecDeque::<anyllm::Result<StreamEvent>>::new(),
            false,
        ),
        move |(mut event_stream, mut state, mut pending, mut finalized)| async move {
            loop {
                if let Some(event) = pending.pop_front() {
                    return Some((event, (event_stream, state, pending, finalized)));
                }

                if finalized {
                    return None;
                }

                match event_stream.next().await {
                    Some(Ok(event)) => pending.extend(state.process_sse_data(&event.data)),
                    Some(Err(e)) => {
                        pending.extend(state.finalize_incomplete());
                        pending.push_back(Err(map_stream_error(e)));
                        finalized = true;
                    }
                    None => {
                        let saw_response = state.started_response;
                        pending.extend(state.finalize_incomplete());
                        if saw_response {
                            pending.push_back(Err(anyllm::Error::Stream(
                                "SSE stream ended before [DONE]".into(),
                            )));
                        }
                        finalized = true;
                    }
                }
            }
        },
    ))
}
