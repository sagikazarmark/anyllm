use std::collections::{HashSet, VecDeque};

use anyllm::{
    ChatStream, ExtraMap, Result, StreamBlockType, StreamEvent, Usage, UsageMetadataMode,
};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde::Deserialize;

use crate::error::{map_api_error, map_stream_error};
use crate::wire;

struct SseState {
    started_response: bool,
    response_id: Option<String>,
    model: Option<String>,
    blocks: HashSet<usize>,
}

fn image_block_data(source: &wire::ImageSource) -> ExtraMap {
    let mut data = ExtraMap::new();
    data.insert(
        "source".to_string(),
        serde_json::to_value(source).expect("Anthropic image source serialization should succeed"),
    );
    data
}

impl SseState {
    fn new() -> Self {
        Self {
            started_response: false,
            response_id: None,
            model: None,
            blocks: HashSet::new(),
        }
    }

    fn ensure_response_started(&mut self, events: &mut Vec<Result<StreamEvent>>) {
        if self.started_response {
            return;
        }

        self.started_response = true;
        events.push(Ok(StreamEvent::ResponseStart {
            id: self.response_id.clone(),
            model: self.model.clone(),
        }));
    }

    fn block_type_for(content_type: &str) -> Option<StreamBlockType> {
        match content_type {
            "text" => Some(StreamBlockType::Text),
            "image" => Some(StreamBlockType::Image),
            "tool_use" => Some(StreamBlockType::ToolCall),
            "thinking" => Some(StreamBlockType::Reasoning),
            _ => None,
        }
    }

    fn close_blocks(&mut self, events: &mut Vec<Result<StreamEvent>>) {
        let mut blocks: Vec<_> = self.blocks.drain().collect();
        blocks.sort_unstable();
        for index in blocks {
            events.push(Ok(StreamEvent::BlockStop { index }));
        }
    }

    fn finish_complete(&mut self, events: &mut Vec<Result<StreamEvent>>) {
        self.close_blocks(events);
        if self.started_response {
            events.push(Ok(StreamEvent::ResponseStop));
            self.started_response = false;
        }
    }

    fn finish_incomplete(&mut self, events: &mut Vec<Result<StreamEvent>>) {
        self.close_blocks(events);
        self.started_response = false;
    }
}

#[derive(Deserialize)]
struct MessageStartEnvelope {
    message: MessageStartPayload,
}

#[derive(Deserialize)]
struct MessageStartPayload {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<MessageStartUsage>,
}

#[derive(Deserialize)]
struct MessageStartUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct ContentBlockStartEnvelope {
    index: usize,
    content_block: ContentBlockStartPayload,
}

#[derive(Deserialize)]
struct ContentBlockStartPayload {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    source: Option<wire::ImageSource>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize)]
struct ContentBlockDeltaEnvelope {
    index: usize,
    delta: ContentBlockDeltaPayload,
}

#[derive(Deserialize)]
struct ContentBlockDeltaPayload {
    #[serde(rename = "type")]
    delta_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    signature: Option<String>,
}

#[derive(Deserialize)]
struct ContentBlockStopEnvelope {
    index: usize,
}

#[derive(Deserialize)]
struct MessageDeltaEnvelope {
    #[serde(default)]
    delta: Option<MessageDeltaPayload>,
    #[serde(default)]
    usage: Option<MessageDeltaUsage>,
}

#[derive(Deserialize)]
struct MessageDeltaPayload {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct MessageDeltaUsage {
    #[serde(default)]
    output_tokens: Option<u64>,
}

fn process_sse_event(
    state: &mut SseState,
    event_type: &str,
    data: &str,
) -> Vec<Result<StreamEvent>> {
    if data.is_empty() {
        return Vec::new();
    }

    let json: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(e) => {
            return vec![Err(anyllm::Error::Stream(format!("SSE parse error: {e}")))];
        }
    };

    let mut events = Vec::new();

    match event_type {
        "message_start" => {
            if let Ok(envelope) = serde_json::from_value::<MessageStartEnvelope>(json) {
                state.response_id = envelope.message.id;
                state.model = envelope.message.model;
                state.ensure_response_started(&mut events);

                if let Some(usage) = envelope.message.usage {
                    // Anthropic's message_start usage is a cumulative snapshot of
                    // prompt/cache-side usage known at stream start.
                    events.push(Ok(StreamEvent::ResponseMetadata {
                        finish_reason: None,
                        usage: Some({
                            let mut usage_stats = Usage::new();
                            usage_stats.input_tokens = usage.input_tokens;
                            usage_stats.cached_input_tokens = usage.cache_read_input_tokens;
                            usage_stats.cache_creation_input_tokens =
                                usage.cache_creation_input_tokens;
                            usage_stats
                        }),
                        usage_mode: UsageMetadataMode::Snapshot,
                        id: state.response_id.clone(),
                        model: state.model.clone(),
                        metadata: ExtraMap::new(),
                    }));
                }
            }
        }
        "content_block_start" => {
            state.ensure_response_started(&mut events);
            if let Ok(envelope) = serde_json::from_value::<ContentBlockStartEnvelope>(json)
                && let Some(stream_type) =
                    SseState::block_type_for(&envelope.content_block.block_type)
            {
                state.blocks.insert(envelope.index);
                let (id, name) = match stream_type {
                    StreamBlockType::ToolCall => {
                        (envelope.content_block.id, envelope.content_block.name)
                    }
                    _ => (None, None),
                };
                let data = match stream_type {
                    StreamBlockType::Image => {
                        envelope.content_block.source.as_ref().map(image_block_data)
                    }
                    _ => None,
                };
                events.push(Ok(StreamEvent::BlockStart {
                    index: envelope.index,
                    block_type: stream_type,
                    id,
                    name,
                    type_name: None,
                    data,
                }));
            }
        }
        "content_block_delta" => {
            state.ensure_response_started(&mut events);
            if let Ok(envelope) = serde_json::from_value::<ContentBlockDeltaEnvelope>(json) {
                match envelope.delta.delta_type.as_str() {
                    "text_delta" => {
                        let text = envelope.delta.text.unwrap_or_default();
                        events.push(Ok(StreamEvent::TextDelta {
                            index: envelope.index,
                            text,
                        }));
                    }
                    "input_json_delta" => {
                        let partial = envelope.delta.partial_json.unwrap_or_default();
                        events.push(Ok(StreamEvent::ToolCallDelta {
                            index: envelope.index,
                            arguments: partial,
                        }));
                    }
                    "thinking_delta" => {
                        let text = envelope.delta.thinking.unwrap_or_default();
                        events.push(Ok(StreamEvent::ReasoningDelta {
                            index: envelope.index,
                            text,
                            signature: None,
                        }));
                    }
                    "signature_delta" => {
                        if let Some(signature) = envelope.delta.signature {
                            events.push(Ok(StreamEvent::ReasoningDelta {
                                index: envelope.index,
                                text: String::new(),
                                signature: Some(signature),
                            }));
                        }
                    }
                    _ => {}
                }
            }
        }
        "content_block_stop" => {
            state.ensure_response_started(&mut events);
            if let Ok(envelope) = serde_json::from_value::<ContentBlockStopEnvelope>(json)
                && state.blocks.remove(&envelope.index)
            {
                events.push(Ok(StreamEvent::BlockStop {
                    index: envelope.index,
                }));
            }
        }
        "message_delta" => {
            state.ensure_response_started(&mut events);
            if let Ok(envelope) = serde_json::from_value::<MessageDeltaEnvelope>(json) {
                let finish_reason = envelope
                    .delta
                    .and_then(|delta| delta.stop_reason)
                    .as_deref()
                    .map(wire::parse_stop_reason);

                let usage = envelope.usage.map(|usage| {
                    let mut usage_stats = Usage::new();
                    usage_stats.output_tokens = usage.output_tokens;
                    usage_stats
                });

                if finish_reason.is_some() || usage.is_some() {
                    // Anthropic message_delta usage is incremental output-side
                    // usage layered onto the earlier message_start snapshot.
                    events.push(Ok(StreamEvent::ResponseMetadata {
                        finish_reason,
                        usage,
                        usage_mode: UsageMetadataMode::Delta,
                        id: state.response_id.clone(),
                        model: state.model.clone(),
                        metadata: ExtraMap::new(),
                    }));
                }
            }
        }
        "message_stop" => {
            state.finish_complete(&mut events);
        }
        "ping" => {}
        "error" => {
            if let Ok(err) = serde_json::from_value::<wire::ErrorResponse>(json) {
                events.push(Err(map_api_error(&err.error)));
            } else {
                events.push(Err(anyllm::Error::Stream("Unknown stream error".into())));
            }
        }
        _ => {}
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
            SseState::new(),
            VecDeque::<Result<StreamEvent>>::new(),
            false,
        ),
        |(mut event_stream, mut state, mut pending, mut finalized)| async move {
            loop {
                if let Some(event) = pending.pop_front() {
                    return Some((event, (event_stream, state, pending, finalized)));
                }

                if finalized {
                    return None;
                }

                match event_stream.next().await {
                    Some(Ok(event)) => {
                        pending.extend(process_sse_event(&mut state, &event.event, &event.data));
                    }
                    Some(Err(e)) => {
                        let mut events = Vec::new();
                        state.finish_incomplete(&mut events);
                        pending.extend(events);
                        pending.push_back(Err(map_stream_error(e)));
                        finalized = true;
                    }
                    None => {
                        let saw_response = state.started_response;
                        let mut events = Vec::new();
                        state.finish_incomplete(&mut events);
                        pending.extend(events);
                        if saw_response {
                            pending.push_back(Err(anyllm::Error::Stream(
                                "SSE stream ended before message_stop".into(),
                            )));
                        }
                        finalized = true;
                    }
                }
            }
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyllm::{ChatStreamExt, FinishReason, StreamBlockType, StreamEvent};

    fn parse_sse_to_events(sse_text: &str) -> Vec<Result<StreamEvent>> {
        let mut state = SseState::new();
        let mut events = Vec::new();

        for block in sse_text.split("\n\n") {
            let block = block.trim();
            if block.is_empty() {
                continue;
            }
            let mut event_type = String::new();
            let mut data = String::new();
            for line in block.lines() {
                if let Some(rest) = line.strip_prefix("event: ") {
                    event_type = rest.to_string();
                } else if let Some(rest) = line.strip_prefix("data: ") {
                    data = rest.to_string();
                }
            }

            events.extend(process_sse_event(&mut state, &event_type, &data));
        }

        events
    }

    #[test]
    fn text_only_stream() {
        let sse_data = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-20250514\",\"stop_reason\":null,\"usage\":{\"input_tokens\":100,\"output_tokens\":0}}}\n\
\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\
\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\
\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":42}}\n\
\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\
\n";

        let events = parse_sse_to_events(sse_data);

        assert!(matches!(
            events[0].as_ref().unwrap(),
            StreamEvent::ResponseStart { id, model }
                if id.as_deref() == Some("msg_1")
                    && model.as_deref() == Some("claude-sonnet-4-20250514")
        ));
        assert!(matches!(
            events[1].as_ref().unwrap(),
            StreamEvent::ResponseMetadata { usage: Some(_), .. }
        ));
        assert!(matches!(
            events[2].as_ref().unwrap(),
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
        assert_eq!(texts, vec!["Hello", " world"]);

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
    fn tool_use_stream() {
        let sse_data = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-20250514\",\"stop_reason\":null,\"usage\":{\"input_tokens\":100,\"output_tokens\":0}}}\n\
\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"read_file\",\"input\":{}}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\": \\\"foo\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\".txt\\\"}\"}}\n\
\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\
\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":30}}\n\
\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\
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
                assert_eq!(id.as_deref(), Some("toolu_1"));
                assert_eq!(name.as_deref(), Some("read_file"));
            }
            other => panic!("expected ToolCallStart, got {other:?}"),
        }

        let deltas: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter(|e| matches!(e, StreamEvent::ToolCallDelta { .. }))
            .collect();
        assert_eq!(deltas.len(), 2);

        assert!(
            events
                .iter()
                .filter_map(|e| e.as_ref().ok())
                .any(|e| matches!(e, StreamEvent::BlockStop { index } if *index == 0))
        );

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
    fn thinking_stream() {
        let sse_data = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-20250514\",\"stop_reason\":null,\"usage\":{\"input_tokens\":50,\"output_tokens\":0}}}\n\
\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let me think...\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_abc\"}}\n\
\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\
\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Answer\"}}\n\
\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":1}\n\
\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":20}}\n\
\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\
\n";

        let events = parse_sse_to_events(sse_data);

        let reasoning: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter(|e| matches!(e, StreamEvent::ReasoningDelta { .. }))
            .collect();
        assert_eq!(reasoning.len(), 2);
        match reasoning[0] {
            StreamEvent::ReasoningDelta {
                text, signature, ..
            } => {
                assert_eq!(text, "Let me think...");
                assert!(signature.is_none());
            }
            other => panic!("expected ReasoningDelta, got {other:?}"),
        }
        match reasoning[1] {
            StreamEvent::ReasoningDelta {
                text, signature, ..
            } => {
                assert!(text.is_empty());
                assert_eq!(signature.as_deref(), Some("sig_abc"));
            }
            other => panic!("expected ReasoningDelta, got {other:?}"),
        }
    }

    #[test]
    fn image_stream() {
        let sse_data = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_img\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-20250514\",\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\
\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"image\",\"source\":{\"type\":\"url\",\"url\":\"https://example.com/cat.png\"}}}\n\
\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\
\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\
\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\
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
                assert_eq!(block_type, &StreamBlockType::Image);
                assert_eq!(
                    data.get("source"),
                    Some(&serde_json::json!({
                        "type": "url",
                        "url": "https://example.com/cat.png"
                    }))
                );
            }
            other => panic!("expected image block start, got {other:?}"),
        }
    }

    #[test]
    fn error_in_stream_overloaded_maps_to_typed_error() {
        let sse_data = "\
event: error\n\
data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        assert_eq!(events.len(), 1);
        let err = events[0].as_ref().unwrap_err();
        match err {
            anyllm::Error::Overloaded { message, .. } => assert_eq!(message, "Overloaded"),
            other => panic!("expected Overloaded error, got {other:?}"),
        }
    }

    #[test]
    fn error_in_stream_rate_limited_maps_to_typed_error() {
        let sse_data = "\
event: error\n\
data: {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"Too many requests\"}}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        let err = events[0].as_ref().unwrap_err();
        match err {
            anyllm::Error::RateLimited { message, .. } => assert_eq!(message, "Too many requests"),
            other => panic!("expected RateLimited error, got {other:?}"),
        }
    }

    #[test]
    fn error_in_stream_auth_maps_to_typed_error() {
        let sse_data = "\
event: error\n\
data: {\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"invalid x-api-key\"}}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        let err = events[0].as_ref().unwrap_err();
        match err {
            anyllm::Error::Auth(msg) => assert_eq!(msg, "invalid x-api-key"),
            other => panic!("expected Auth error, got {other:?}"),
        }
    }

    #[test]
    fn error_in_stream_unparseable_falls_back_to_stream_error() {
        let sse_data = "\
event: error\n\
data: {\"not_an_error\": true}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        let err = events[0].as_ref().unwrap_err();
        assert!(matches!(err, anyllm::Error::Stream(_)));
    }

    #[test]
    fn cache_tokens_in_message_start() {
        let sse_data = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-20250514\",\"stop_reason\":null,\"usage\":{\"input_tokens\":100,\"output_tokens\":0,\"cache_read_input_tokens\":80,\"cache_creation_input_tokens\":15}}}\n\
\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\
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
        assert_eq!(usage.cached_input_tokens, Some(80));
        assert_eq!(usage.cache_creation_input_tokens, Some(15));
    }

    #[test]
    fn message_start_usage_is_snapshot() {
        let sse_data = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-20250514\",\"stop_reason\":null,\"usage\":{\"input_tokens\":100}}}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        assert!(matches!(
            events.iter().find_map(|e| e.as_ref().ok()),
            Some(StreamEvent::ResponseStart { .. })
        ));
        let metadata = events
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
        assert_eq!(metadata, UsageMetadataMode::Snapshot);
    }

    #[test]
    fn message_delta_usage_is_delta() {
        let sse_data = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-20250514\",\"stop_reason\":null,\"usage\":{\"input_tokens\":100}}}\n\
\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":42}}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        let modes: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter_map(|e| match e {
                StreamEvent::ResponseMetadata {
                    usage_mode,
                    usage: Some(_),
                    ..
                } => Some(*usage_mode),
                _ => None,
            })
            .collect();
        assert_eq!(
            modes,
            vec![UsageMetadataMode::Snapshot, UsageMetadataMode::Delta]
        );
    }

    #[tokio::test]
    async fn sse_to_stream_preserves_reasoning_signature() {
        let sse_data = b"\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-20250514\",\"stop_reason\":null,\"usage\":{\"input_tokens\":50,\"output_tokens\":0}}}\n\
\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let me think...\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_abc\"}}\n\
\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\
\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":20}}\n\
\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\
\n";

        let byte_stream =
            futures_util::stream::iter(vec![Ok::<&[u8], std::io::Error>(sse_data.as_slice())]);
        let chat_stream = sse_to_stream(byte_stream);
        let response = chat_stream.collect_response().await.unwrap();

        assert_eq!(response.reasoning_text(), Some("Let me think...".into()));
        match &response.content[0] {
            anyllm::ContentBlock::Reasoning { signature, .. } => {
                assert_eq!(signature.as_deref(), Some("sig_abc"));
            }
            other => panic!("expected Reasoning, got {other:?}"),
        }
    }
}
