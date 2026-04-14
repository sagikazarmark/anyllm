use anyllm::UserContent;
use anyllm::{
    ChatRequest, ContentBlock, ContentPart, ImageSource, Message, StreamEvent, Tool, ToolChoice,
};
use anyllm_openai_compat::{
    RequestOptions, SseState, process_sse_data, to_chat_completion_request,
};
use std::hint::black_box;
use criterion::{Criterion, criterion_group, criterion_main};
use serde_json::json;

fn bench_request() -> ChatRequest {
    let mut request = ChatRequest::new("gpt-4o-mini")
        .system("You are a structured assistant.")
        .message(Message::User {
            content: UserContent::Parts(vec![
                ContentPart::Text {
                    text: "Summarize this image and text payload.".into(),
                },
                ContentPart::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/image.png".into(),
                    },
                    detail: Some("high".into()),
                },
            ]),
            name: Some("bench-user".into()),
            extensions: None,
        })
        .message(Message::Assistant {
            content: vec![
                ContentBlock::Text {
                    text: "I can help with that. ".repeat(8),
                },
                ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "search_docs".into(),
                    arguments: json!({"query": "streaming API design", "limit": 5}).to_string(),
                },
            ],
            name: Some("bench-assistant".into()),
            extensions: None,
        })
        .message(Message::Tool {
            tool_call_id: "call_1".into(),
            name: "search_docs".into(),
            content: "ok".into(),
            is_error: None,
            extensions: None,
        })
        .temperature(0.2)
        .max_tokens(512)
        .tool_choice(ToolChoice::Auto)
        .tools([
            Tool::new(
                "search_docs",
                json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"}
                    },
                    "required": ["query"]
                }),
            )
            .description("Search internal docs"),
            Tool::new(
                "write_note",
                json!({
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"}
                    },
                    "required": ["title", "body"]
                }),
            )
            .description("Persist a note"),
        ]);

    request.parallel_tool_calls = Some(true);
    request
}

fn sse_chunks() -> Vec<String> {
    vec![
        json!({
            "id": "chatcmpl-bench",
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": null}]
        })
        .to_string(),
        json!({
            "id": "chatcmpl-bench",
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "search_docs", "arguments": "{\"query\":"}}]}, "finish_reason": null}]
        })
        .to_string(),
        json!({
            "id": "chatcmpl-bench",
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"rust\"}"}}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240}
        })
        .to_string(),
        "[DONE]".to_string(),
    ]
}

fn bench_request_normalization(c: &mut Criterion) {
    let request = bench_request();
    let options = RequestOptions::default();

    c.bench_function("openai_compat_to_chat_completion_request", |b| {
        b.iter(|| {
            black_box(to_chat_completion_request(
                black_box(&request),
                true,
                black_box(&options),
            ))
            .unwrap()
        })
    });
}

fn bench_sse_processing(c: &mut Criterion) {
    let chunks = sse_chunks();

    c.bench_function("openai_compat_process_sse_sequence", |b| {
        b.iter(|| {
            let mut state = SseState::new();
            let mut events = Vec::<anyllm::Result<StreamEvent>>::new();
            for chunk in black_box(&chunks) {
                events.extend(process_sse_data(&mut state, chunk));
            }
            black_box(events)
        })
    });
}

criterion_group!(benches, bench_request_normalization, bench_sse_processing);
criterion_main!(benches);
