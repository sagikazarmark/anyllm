use anyllm::ChatStream;
#[cfg(test)]
use anyllm::Result;

use crate::error::map_stream_error;

pub(crate) fn sse_to_stream<S, B, E>(byte_stream: S) -> ChatStream
where
    S: futures::Stream<Item = std::result::Result<B, E>> + Send + Unpin + 'static,
    B: AsRef<[u8]>,
    E: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
{
    anyllm_openai_compat::sse_to_stream(byte_stream, map_stream_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyllm::{ChatStreamExt, FinishReason, StreamBlockType, StreamEvent};
    use anyllm_openai_compat::{SseState, process_sse_data};

    fn parse_sse_to_events(sse_text: &str) -> Vec<Result<StreamEvent>> {
        let mut state = SseState::new();
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

        events
    }

    #[test]
    fn text_only_stream() {
        let sse_data = "\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\
\n\
data: [DONE]\n\
\n";

        let events = parse_sse_to_events(sse_data);

        assert!(matches!(
            events[0].as_ref().unwrap(),
            StreamEvent::ResponseStart { id, model }
                if id.as_deref() == Some("chatcmpl-abc") && model.as_deref() == Some("gpt-4o")
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
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(5));

        assert!(matches!(
            events.last().unwrap().as_ref().unwrap(),
            StreamEvent::ResponseStop
        ));
    }

    #[test]
    fn tool_use_stream() {
        let sse_data = "\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\"\"}}]},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\": \\\"foo.txt\\\"}\"}}]},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\
\n\
data: [DONE]\n\
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
                assert_eq!(*index, 1);
                assert_eq!(id.as_deref(), Some("call_abc"));
                assert_eq!(name.as_deref(), Some("read_file"));
            }
            other => panic!("expected tool block start, got {other:?}"),
        }

        let deltas: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter(|e| matches!(e, StreamEvent::ToolCallDelta { .. }))
            .collect();
        assert_eq!(deltas.len(), 2);

        let stops: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter(|e| matches!(e, StreamEvent::BlockStop { index } if *index == 1))
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
    fn parallel_tool_calls_stream() {
        let sse_data = "\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}}]},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_2\",\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"{\\\"path\\\":\\\"b.txt\\\"}\"}}]},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\
\n\
data: [DONE]\n\
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
        assert_eq!(starts, vec![1, 2]);

        let stops: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter_map(|e| match e {
                StreamEvent::BlockStop { index } if *index > 0 => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(stops, vec![1, 2]);
    }

    #[test]
    fn mixed_text_and_tool_stream() {
        let sse_data = "\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Let me \"},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"check.\"},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"q\\\":\\\"test\\\"}\"}}]},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\
\n\
data: [DONE]\n\
\n";

        let events = parse_sse_to_events(sse_data);

        let texts: Vec<_> = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .filter_map(|e| match e {
                StreamEvent::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Let me ", "check."]);

        let text_stop_before_tool = events
            .iter()
            .position(|e| matches!(e.as_ref().unwrap(), StreamEvent::BlockStop { index: 0 }))
            .unwrap();
        let tool_start = events
            .iter()
            .position(|e| {
                matches!(
                    e.as_ref().unwrap(),
                    StreamEvent::BlockStart {
                        block_type: StreamBlockType::ToolCall,
                        ..
                    }
                )
            })
            .unwrap();
        assert!(text_stop_before_tool < tool_start);
    }

    #[test]
    fn done_sentinel_produces_response_stop() {
        let mut state = SseState::new();
        let _ = process_sse_data(
            &mut state,
            r#"{"id":"resp_1","model":"gpt-4o","choices":[]}"#,
        );
        let results = process_sse_data(&mut state, "[DONE]");
        assert!(matches!(
            results.as_slice(),
            [Ok(StreamEvent::ResponseStop)]
        ));
    }

    #[test]
    fn malformed_json_produces_error() {
        let mut state = SseState::new();
        let results = process_sse_data(&mut state, "{not valid json");
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
        assert!(matches!(results[0], Err(anyllm::Error::Stream(_))));
    }

    #[test]
    fn usage_with_cached_and_reasoning_tokens() {
        let sse_data = "\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":50,\"total_tokens\":150,\"prompt_tokens_details\":{\"cached_tokens\":80},\"completion_tokens_details\":{\"reasoning_tokens\":20}}}\n\
\n";

        let events = parse_sse_to_events(sse_data);
        assert_eq!(events.len(), 2);
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
        assert_eq!(usage.total_tokens, Some(150));
        assert_eq!(usage.cached_input_tokens, Some(80));
        assert_eq!(usage.reasoning_tokens, Some(20));
    }

    #[tokio::test]
    async fn sse_to_stream_tool_use() {
        let sse_data = b"\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\": \\\"foo.txt\\\"}\"}}]},\"finish_reason\":null}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\
\n\
data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":30,\"total_tokens\":80}}\n\
\n\
data: [DONE]\n\
\n";

        let byte_stream =
            futures::stream::iter(vec![Ok::<&[u8], std::io::Error>(sse_data.as_slice())]);
        let chat_stream = sse_to_stream(byte_stream);
        let response = chat_stream.collect_response().await.unwrap();

        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert!(response.has_tool_calls());
        let calls: Vec<_> = response.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].name, "read_file");
        let args: serde_json::Value = serde_json::from_str(calls[0].arguments).unwrap();
        assert_eq!(args["path"], "foo.txt");
    }
}
