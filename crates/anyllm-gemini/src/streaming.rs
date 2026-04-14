use std::collections::VecDeque;

use anyllm::{
    ChatStream, ContentBlock, ExtraMap, FinishReason, ImageSource, Result, StreamBlockType,
    StreamEvent, UsageMetadataMode,
};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use uuid::Uuid;

use crate::error::map_stream_error;
use crate::wire::{
    Candidate, GenerateContentResponse, from_usage_metadata, parse_finish_reason,
    synthetic_tool_call_id,
};

struct TrackedBlock {
    content: ContentBlock,
    open: bool,
}

struct StreamState {
    started_response: bool,
    response_id: Option<String>,
    model: Option<String>,
    anonymous_tool_call_scope: String,
    blocks: Vec<TrackedBlock>,
}

impl StreamState {
    fn new() -> Self {
        Self {
            started_response: false,
            response_id: None,
            model: None,
            anonymous_tool_call_scope: Uuid::new_v4().to_string(),
            blocks: Vec::new(),
        }
    }

    fn ensure_response_started(
        &mut self,
        chunk: &GenerateContentResponse,
        events: &mut Vec<Result<StreamEvent>>,
    ) {
        if let Some(id) = &chunk.response_id {
            self.response_id = Some(id.clone());
        }
        if let Some(model) = &chunk.model_version {
            self.model = Some(model.clone());
        }

        if self.started_response {
            return;
        }

        self.started_response = true;
        events.push(Ok(StreamEvent::ResponseStart {
            id: self.response_id.clone(),
            model: self.model.clone(),
        }));
    }

    fn close_open_blocks(&mut self, events: &mut Vec<Result<StreamEvent>>) {
        for (index, block) in self.blocks.iter_mut().enumerate() {
            if block.open {
                block.open = false;
                events.push(Ok(StreamEvent::BlockStop { index }));
            }
        }
    }

    fn finalize(&mut self) -> Vec<Result<StreamEvent>> {
        let mut events = Vec::new();
        self.close_open_blocks(&mut events);
        if self.started_response {
            events.push(Ok(StreamEvent::ResponseStop));
            self.started_response = false;
        }
        events
    }
}

fn image_block_data(source: &ImageSource) -> ExtraMap {
    let mut data = ExtraMap::new();
    data.insert(
        "source".to_string(),
        serde_json::to_value(source).expect("Gemini image source serialization should succeed"),
    );
    data
}

fn normalized_candidate_content(
    candidate: Candidate,
    response_id: Option<&str>,
    anonymous_tool_call_scope: &str,
) -> Result<(Vec<ContentBlock>, Option<FinishReason>)> {
    let raw_finish_reason = candidate.finish_reason.as_deref().map(parse_finish_reason);
    let mut blocks = Vec::new();
    let mut tool_call_index = 0usize;

    if let Some(content) = candidate.content {
        for part in content.parts {
            if let Some(function_call) = part.function_call {
                let arguments =
                    serde_json::to_string(&function_call.args).map_err(anyllm::Error::from)?;
                blocks.push(ContentBlock::ToolCall {
                    id: synthetic_tool_call_id(
                        response_id,
                        Some(anonymous_tool_call_scope),
                        tool_call_index,
                    ),
                    name: function_call.name,
                    arguments,
                });
                tool_call_index += 1;
            } else if let Some(inline_data) = part.inline_data {
                blocks.push(ContentBlock::Image {
                    source: ImageSource::Base64 {
                        media_type: inline_data.mime_type,
                        data: inline_data.data,
                    },
                });
            } else if let Some(file_data) = part.file_data {
                blocks.push(ContentBlock::Image {
                    source: ImageSource::Url {
                        url: file_data.file_uri,
                    },
                });
            } else if let Some(text) = part.text
                && !text.is_empty()
            {
                if part.thought == Some(true) {
                    blocks.push(ContentBlock::Reasoning {
                        text,
                        signature: None,
                    });
                } else {
                    blocks.push(ContentBlock::Text { text });
                }
            }
        }
    }

    let finish_reason = if blocks
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolCall { .. }))
    {
        Some(FinishReason::ToolCalls)
    } else {
        raw_finish_reason
    };

    Ok((blocks, finish_reason))
}

fn emit_appended_suffix(
    previous: &str,
    current: &str,
    kind: &str,
    index: usize,
) -> Result<Option<String>> {
    if current == previous {
        return Ok(None);
    }

    let Some(suffix) = current.strip_prefix(previous) else {
        return Err(anyllm::Error::Stream(format!(
            "Gemini {kind} block at index {index} was not append-only"
        )));
    };

    Ok(Some(suffix.to_string()))
}

fn emit_new_block(
    index: usize,
    block: &ContentBlock,
    events: &mut Vec<Result<StreamEvent>>,
) -> Result<bool> {
    match block {
        ContentBlock::Text { text } => {
            events.push(Ok(StreamEvent::BlockStart {
                index,
                block_type: StreamBlockType::Text,
                id: None,
                name: None,
                type_name: None,
                data: None,
            }));
            events.push(Ok(StreamEvent::TextDelta {
                index,
                text: text.clone(),
            }));
            Ok(true)
        }
        ContentBlock::Reasoning { text, signature } => {
            events.push(Ok(StreamEvent::BlockStart {
                index,
                block_type: StreamBlockType::Reasoning,
                id: None,
                name: None,
                type_name: None,
                data: None,
            }));
            events.push(Ok(StreamEvent::ReasoningDelta {
                index,
                text: text.clone(),
                signature: signature.clone(),
            }));
            Ok(true)
        }
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
        } => {
            events.push(Ok(StreamEvent::BlockStart {
                index,
                block_type: StreamBlockType::ToolCall,
                id: Some(id.clone()),
                name: Some(name.clone()),
                type_name: None,
                data: None,
            }));
            if !arguments.is_empty() {
                events.push(Ok(StreamEvent::ToolCallDelta {
                    index,
                    arguments: arguments.clone(),
                }));
            }
            Ok(true)
        }
        ContentBlock::Image { source } => {
            events.push(Ok(StreamEvent::BlockStart {
                index,
                block_type: StreamBlockType::Image,
                id: None,
                name: None,
                type_name: None,
                data: Some(image_block_data(source)),
            }));
            events.push(Ok(StreamEvent::BlockStop { index }));
            Ok(false)
        }
        ContentBlock::Other { type_name, .. } => Err(anyllm::Error::Stream(format!(
            "Gemini stream emitted unsupported block type '{type_name}' at index {index}"
        ))),
        _ => Err(anyllm::Error::Stream(format!(
            "Gemini stream emitted an unknown block variant at index {index}"
        ))),
    }
}

fn emit_existing_block_delta(
    index: usize,
    previous: &TrackedBlock,
    current: &ContentBlock,
    events: &mut Vec<Result<StreamEvent>>,
) -> Result<()> {
    if !previous.open && current != &previous.content {
        return Err(anyllm::Error::Stream(format!(
            "Gemini updated closed block at index {index}"
        )));
    }

    match (&previous.content, current) {
        (ContentBlock::Text { text: previous }, ContentBlock::Text { text: current }) => {
            if let Some(text) = emit_appended_suffix(previous, current, "text", index)? {
                events.push(Ok(StreamEvent::TextDelta { index, text }));
            }
            Ok(())
        }
        (
            ContentBlock::Reasoning {
                text: previous_text,
                signature: previous_signature,
            },
            ContentBlock::Reasoning {
                text: current_text,
                signature: current_signature,
            },
        ) => {
            let new_text = emit_appended_suffix(previous_text, current_text, "reasoning", index)?;
            if let Some(text) = new_text {
                events.push(Ok(StreamEvent::ReasoningDelta {
                    index,
                    text,
                    signature: current_signature.clone(),
                }));
            } else if previous_signature != current_signature {
                events.push(Ok(StreamEvent::ReasoningDelta {
                    index,
                    text: String::new(),
                    signature: current_signature.clone(),
                }));
            }
            Ok(())
        }
        (
            ContentBlock::ToolCall {
                id: previous_id,
                name: previous_name,
                arguments: previous_arguments,
            },
            ContentBlock::ToolCall {
                id: current_id,
                name: current_name,
                arguments: current_arguments,
            },
        ) => {
            if previous_id != current_id || previous_name != current_name {
                return Err(anyllm::Error::Stream(format!(
                    "Gemini changed tool call identity at index {index}"
                )));
            }
            if let Some(arguments) =
                emit_appended_suffix(previous_arguments, current_arguments, "tool call", index)?
            {
                events.push(Ok(StreamEvent::ToolCallDelta { index, arguments }));
            }
            Ok(())
        }
        (ContentBlock::Image { source: previous }, ContentBlock::Image { source: current })
            if previous == current =>
        {
            Ok(())
        }
        _ => Err(anyllm::Error::Stream(format!(
            "Gemini changed block type or identity at index {index}"
        ))),
    }
}

fn process_sse_data(state: &mut StreamState, data: &str) -> Vec<Result<StreamEvent>> {
    if data.is_empty() {
        return Vec::new();
    }

    let chunk: GenerateContentResponse = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(e) => {
            return vec![Err(anyllm::Error::Stream(format!("SSE parse error: {e}")))];
        }
    };

    if let Some(err) = chunk
        .prompt_feedback
        .as_ref()
        .and_then(crate::wire::prompt_feedback_error)
    {
        return vec![Err(err)];
    }

    let mut events: Vec<Result<StreamEvent>> = Vec::new();
    state.ensure_response_started(&chunk, &mut events);

    let candidate = match chunk.candidates.into_iter().next() {
        Some(c) => c,
        None => {
            if let Some(meta) = chunk.usage_metadata {
                // Gemini reports cumulative usage snapshots for the response.
                events.push(Ok(StreamEvent::ResponseMetadata {
                    finish_reason: None,
                    usage: Some(from_usage_metadata(meta)),
                    usage_mode: UsageMetadataMode::Snapshot,
                    id: state.response_id.clone(),
                    model: state.model.clone(),
                    metadata: ExtraMap::new(),
                }));
            }
            return events;
        }
    };

    let (blocks, finish_reason) = match normalized_candidate_content(
        candidate,
        state.response_id.as_deref(),
        &state.anonymous_tool_call_scope,
    ) {
        Ok(value) => value,
        Err(err) => {
            events.push(Err(err));
            return events;
        }
    };

    if blocks.len() < state.blocks.len() {
        events.push(Err(anyllm::Error::Stream(format!(
            "Gemini streamed content shrank from {} blocks to {}",
            state.blocks.len(),
            blocks.len()
        ))));
        return events;
    }

    let mut next_blocks = Vec::with_capacity(blocks.len());
    for (index, block) in blocks.iter().enumerate() {
        let open = match state.blocks.get(index) {
            Some(previous) => {
                if let Err(err) = emit_existing_block_delta(index, previous, block, &mut events) {
                    events.push(Err(err));
                    return events;
                }
                previous.open
            }
            None => match emit_new_block(index, block, &mut events) {
                Ok(open) => open,
                Err(err) => {
                    events.push(Err(err));
                    return events;
                }
            },
        };

        next_blocks.push(TrackedBlock {
            content: block.clone(),
            open,
        });
    }
    state.blocks = next_blocks;

    if let Some(finish_reason) = finish_reason {
        state.close_open_blocks(&mut events);
        // Gemini finish reasons are terminal response metadata only; usage, when
        // present, arrives separately as a cumulative snapshot.
        events.push(Ok(StreamEvent::ResponseMetadata {
            finish_reason: Some(finish_reason),
            usage: None,
            usage_mode: UsageMetadataMode::Snapshot,
            id: state.response_id.clone(),
            model: state.model.clone(),
            metadata: ExtraMap::new(),
        }));
    }

    if let Some(meta) = chunk.usage_metadata {
        // Gemini usage metadata is cumulative for the response.
        events.push(Ok(StreamEvent::ResponseMetadata {
            finish_reason: None,
            usage: Some(from_usage_metadata(meta)),
            usage_mode: UsageMetadataMode::Snapshot,
            id: state.response_id.clone(),
            model: state.model.clone(),
            metadata: ExtraMap::new(),
        }));
    }

    events
}

pub(crate) fn sse_to_stream<S, B, E>(byte_stream: S) -> ChatStream
where
    S: futures_core::Stream<Item = std::result::Result<B, E>> + Send + Unpin + 'static,
    B: AsRef<[u8]>,
    E: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
{
    let event_stream = byte_stream.eventsource();

    Box::pin(futures_util::stream::unfold(
        (
            event_stream,
            StreamState::new(),
            VecDeque::<Result<StreamEvent>>::new(),
            false,
        ),
        |(mut event_stream, mut state, mut pending, mut finalized)| async move {
            loop {
                if let Some(event) = pending.pop_front() {
                    return Some((event, (event_stream, state, pending, finalized)));
                }

                match event_stream.next().await {
                    Some(Ok(event)) => pending.extend(process_sse_data(&mut state, &event.data)),
                    Some(Err(e)) => {
                        pending.extend(state.finalize());
                        pending.push_back(Err(map_stream_error(e)));
                        finalized = true;
                    }
                    None if !finalized => {
                        finalized = true;
                        pending.extend(state.finalize());
                    }
                    None => return None,
                }
            }
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyllm::{ChatStreamExt, FinishReason, StreamBlockType, StreamEvent};
    use futures_util::StreamExt;

    fn parse_sse_to_events(sse_text: &str) -> Vec<Result<StreamEvent>> {
        let mut state = StreamState::new();
        let mut events = Vec::new();

        for block in sse_text.split("\n\n") {
            let block = block.trim();
            if block.is_empty() {
                continue;
            }
            let mut data = String::new();
            for line in block.lines() {
                if let Some(rest) = line.strip_prefix("data: ") {
                    data = rest.to_string();
                }
            }
            events.extend(process_sse_data(&mut state, &data));
        }

        events.extend(state.finalize());

        events
    }

    #[test]
    fn text_only_stream_accumulated() {
        let sse_data = "\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]}}]}\n\
\n\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello, world\"}]}}]}\n\
\n\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello, world!\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":5,\"totalTokenCount\":15}}\n\
\n";

        let events = parse_sse_to_events(sse_data);

        assert!(matches!(
            events[0].as_ref().unwrap(),
            StreamEvent::ResponseStart { id, model }
                if id.as_deref() == Some("resp_1") && model.as_deref() == Some("gemini-2.5-pro")
        ));
        assert!(matches!(
            events[1].as_ref().unwrap(),
            StreamEvent::BlockStart {
                index: 0,
                block_type: StreamBlockType::Text,
                ..
            }
        ));

        let texts: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter_map(|e| match e {
                StreamEvent::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Hello", ", world", "!"]);

        let finish = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find_map(|e| match e {
                StreamEvent::ResponseMetadata { finish_reason, .. } => finish_reason.clone(),
                _ => None,
            })
            .unwrap();
        assert_eq!(finish, FinishReason::Stop);

        assert!(matches!(
            events.last().unwrap().as_ref().unwrap(),
            StreamEvent::ResponseStop
        ));
    }

    #[test]
    fn tool_call_stream_complete_args_in_single_chunk() {
        let sse_data = "\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"read_file\",\"args\":{\"path\":\"/tmp/test.txt\"}}}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":20,\"candidatesTokenCount\":10,\"totalTokenCount\":30}}\n\
\n";

        let events = parse_sse_to_events(sse_data);

        let starts: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter(|e| {
                matches!(
                    e,
                    StreamEvent::BlockStart {
                        block_type: StreamBlockType::ToolCall,
                        ..
                    }
                )
            })
            .collect();
        assert_eq!(starts.len(), 1);
        match starts[0] {
            StreamEvent::BlockStart {
                index, id, name, ..
            } => {
                assert_eq!(*index, 0);
                assert_eq!(id.as_deref(), Some("gemini:resp_1:call:0"));
                assert_eq!(name.as_deref(), Some("read_file"));
            }
            other => panic!("expected ToolCallStart, got {other:?}"),
        }

        let deltas: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter(|e| matches!(e, StreamEvent::ToolCallDelta { .. }))
            .collect();
        assert_eq!(deltas.len(), 1);

        let stops: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter(|e| matches!(e, StreamEvent::BlockStop { index } if *index == 0))
            .collect();
        assert_eq!(stops.len(), 1);

        let finish = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find_map(|e| match e {
                StreamEvent::ResponseMetadata { finish_reason, .. } => finish_reason.clone(),
                _ => None,
            })
            .unwrap();
        assert_eq!(finish, FinishReason::ToolCalls);
    }

    #[test]
    fn parallel_tool_calls_in_single_chunk() {
        let sse_data = "\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"search\",\"args\":{\"q\":\"rust\"}}},{\"functionCall\":{\"name\":\"search\",\"args\":{\"q\":\"python\"}}}]},\"finishReason\":\"STOP\"}]}\n\
\n";

        let events = parse_sse_to_events(sse_data);

        let starts: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter_map(|e| match e {
                StreamEvent::BlockStart {
                    index,
                    block_type: StreamBlockType::ToolCall,
                    ..
                } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(starts, vec![0, 1]);

        let stops: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter_map(|e| match e {
                StreamEvent::BlockStop { index } if *index <= 1 => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(stops, vec![0, 1]);
    }

    #[tokio::test]
    async fn mixed_content_stream_preserves_provider_block_order() {
        let sse_bytes = b"\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[{\"thought\":true,\"text\":\"Thinking\"},{\"text\":\"Answer: \"},{\"functionCall\":{\"name\":\"lookup\",\"args\":{\"q\":\"rust\"}}},{\"text\":\"done\"}]},\"finishReason\":\"STOP\"}]}\n\
\n";

        let byte_stream =
            futures_util::stream::iter(vec![Ok::<&[u8], std::io::Error>(sse_bytes.as_slice())]);
        let response = sse_to_stream(byte_stream).collect_response().await.unwrap();

        assert!(matches!(
            &response.content[0],
            anyllm::ContentBlock::Reasoning { text, .. } if text == "Thinking"
        ));
        assert!(matches!(
            &response.content[1],
            anyllm::ContentBlock::Text { text } if text == "Answer: "
        ));
        assert!(matches!(
            &response.content[2],
            anyllm::ContentBlock::ToolCall { name, arguments, .. }
                if name == "lookup" && arguments == r#"{"q":"rust"}"#
        ));
        assert!(matches!(
            &response.content[3],
            anyllm::ContentBlock::Text { text } if text == "done"
        ));
    }

    #[test]
    fn thought_parts_become_reasoning_deltas() {
        let sse_data = "\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[{\"thought\":true,\"text\":\"Let me think\"},{\"text\":\"Answer\"}]},\"finishReason\":\"STOP\"}]}\n\
\n";

        let events = parse_sse_to_events(sse_data);

        let reasoning: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter_map(|e| match e {
                StreamEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, vec!["Let me think"]);

        let text: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter_map(|e| match e {
                StreamEvent::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, vec!["Answer"]);
    }

    #[test]
    fn safety_finish_reason_maps_to_content_filter() {
        let sse_data = "\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"SAFETY\"}]}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        let finish = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find_map(|e| match e {
                StreamEvent::ResponseMetadata { finish_reason, .. } => finish_reason.clone(),
                _ => None,
            })
            .unwrap();
        assert_eq!(finish, FinishReason::ContentFilter);
    }

    #[test]
    fn blocked_prompt_feedback_maps_to_content_filtered_error() {
        let sse_data = "\
data: {\"promptFeedback\":{\"blockReason\":\"SAFETY\",\"blockReasonMessage\":\"Prompt blocked by safety filters\"},\"candidates\":[]}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            Err(anyllm::Error::ContentFiltered(message)) if message == "Prompt blocked by safety filters"
        ));
    }

    #[tokio::test]
    async fn eof_without_finish_reason_still_collects_response() {
        let sse_data = b"\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]}}]}\n\
\n";

        let byte_stream =
            futures_util::stream::iter(vec![Ok::<&[u8], std::io::Error>(sse_data.as_slice())]);
        let mut collector = anyllm::StreamCollector::new();
        let mut stream = std::pin::pin!(sse_to_stream(byte_stream));

        while let Some(event) = stream.next().await {
            collector.push(event.unwrap()).unwrap();
        }

        let response = collector.finish().unwrap();
        assert_eq!(response.text(), Some("Hello".into()));
        assert_eq!(response.finish_reason, None);
    }

    #[test]
    fn empty_data_produces_no_events() {
        let mut state = StreamState::new();
        let results = process_sse_data(&mut state, "");
        assert!(results.is_empty());
    }

    #[test]
    fn malformed_json_produces_stream_error() {
        let mut state = StreamState::new();
        let results = process_sse_data(&mut state, "{not valid json");
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0], Err(anyllm::Error::Stream(_))));
    }

    #[test]
    fn usage_only_chunk_emits_usage() {
        let sse_data = "\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[],\"usageMetadata\":{\"promptTokenCount\":100,\"candidatesTokenCount\":50,\"totalTokenCount\":150,\"thoughtsTokenCount\":30}}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        let usage = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find_map(|e| match e {
                StreamEvent::ResponseMetadata {
                    usage: Some(usage), ..
                } => Some(usage),
                _ => None,
            })
            .unwrap();
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(50));
        assert_eq!(usage.reasoning_tokens, Some(30));
    }

    #[test]
    fn usage_only_chunk_uses_snapshot_mode() {
        let sse_data = "\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[],\"usageMetadata\":{\"promptTokenCount\":100,\"candidatesTokenCount\":50,\"totalTokenCount\":150}}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        let mode = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find_map(|e| match e {
                StreamEvent::ResponseMetadata {
                    usage_mode,
                    usage: Some(_),
                    ..
                } => Some(*usage_mode),
                _ => None,
            })
            .unwrap();
        assert_eq!(mode, UsageMetadataMode::Snapshot);
    }

    #[test]
    fn terminal_chunk_usage_uses_snapshot_mode() {
        let sse_data = "\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":5,\"totalTokenCount\":15}}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        let usage_modes: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter_map(|e| match e {
                StreamEvent::ResponseMetadata {
                    usage_mode, usage, ..
                } if usage.is_some() => Some(*usage_mode),
                _ => None,
            })
            .collect();
        assert_eq!(usage_modes, vec![UsageMetadataMode::Snapshot]);
    }

    #[tokio::test]
    async fn truncated_text_stream_collects_after_eof_finalization() {
        let sse_bytes = b"\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]}}]}\n\
\n";

        let byte_stream =
            futures_util::stream::iter(vec![Ok::<&[u8], std::io::Error>(sse_bytes.as_slice())]);
        let chat_stream = sse_to_stream(byte_stream);
        let response = chat_stream.collect_response().await.unwrap();

        assert_eq!(response.text(), Some("Hello".into()));
        assert_eq!(response.finish_reason, None);
    }

    #[tokio::test]
    async fn truncated_tool_call_stream_collects_completed_tool_call() {
        let sse_bytes = b"\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"read_file\",\"args\":{\"path\":\"/etc/hosts\"}}}]}}]}\n\
\n";

        let byte_stream =
            futures_util::stream::iter(vec![Ok::<&[u8], std::io::Error>(sse_bytes.as_slice())]);
        let chat_stream = sse_to_stream(byte_stream);
        let response = chat_stream.collect_response().await.unwrap();
        let calls: Vec<_> = response.tool_calls().collect();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "gemini:resp_1:call:0");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, r#"{"path":"/etc/hosts"}"#);
    }

    #[test]
    fn image_part_emits_image_block() {
        let sse_data = "\
data: {\"responseId\":\"resp_1\",\"modelVersion\":\"gemini-2.5-pro\",\"candidates\":[{\"content\":{\"parts\":[{\"inlineData\":{\"mimeType\":\"image/png\",\"data\":\"iVBORw0KGgoAAAANSUhEUg==\"}}]}}]}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        let image_start = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find(|event| {
                matches!(
                    event,
                    StreamEvent::BlockStart {
                        block_type: StreamBlockType::Image,
                        ..
                    }
                )
            })
            .unwrap();

        match image_start {
            StreamEvent::BlockStart {
                index,
                block_type,
                data: Some(data),
                ..
            } => {
                assert_eq!(*index, 0);
                assert_eq!(*block_type, StreamBlockType::Image);
                assert_eq!(
                    data.get("source"),
                    Some(&serde_json::json!({
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUg=="
                    }))
                );
            }
            other => panic!("expected image block start, got {other:?}"),
        }

        assert!(
            events
                .iter()
                .filter_map(|e| e.as_ref().ok())
                .any(|event| matches!(event, StreamEvent::BlockStop { index } if *index == 0))
        );
    }
}
