use anyllm::{
    ChatRequest, ContentPart, FinishReason, Message, ResponseFormat, StreamCollector, Tool,
    ToolChoice,
};
use anyllm_openai_compat::{
    RequestOptions, SseState, process_sse_data, to_chat_completion_request,
};
use serde_json::json;

#[test]
fn compat_can_power_custom_provider_request_and_stream_translation() {
    let request = ChatRequest::new("custom-chat")
        .system("You are a custom provider.")
        .message(Message::user_multimodal(vec![
            ContentPart::text("Look at this image and search the docs."),
            ContentPart::image_url("https://example.com/cat.png"),
        ]))
        .tools(vec![
            Tool::new(
                "search_docs",
                json!({
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"}
                    },
                    "required": ["q"]
                }),
            )
            .description("Search internal docs")
            .with_extension("strict", json!(true)),
        ])
        .tool_choice(ToolChoice::Required)
        .response_format(ResponseFormat::JsonSchema {
            name: Some("answer".into()),
            schema: json!({
                "type": "object",
                "properties": {
                    "summary": {"type": "string"}
                },
                "required": ["summary"]
            }),
            strict: Some(true),
        })
        .seed(123)
        .parallel_tool_calls(true);

    let provider_options = RequestOptions {
        seed: Some(7),
        parallel_tool_calls: Some(false),
        user: Some("user-42".into()),
        service_tier: Some("priority".into()),
        store: Some(true),
        metadata: Some(serde_json::Map::from_iter([(
            "tenant".to_string(),
            json!("acme"),
        )])),
        extra: serde_json::Map::from_iter([
            ("provider_mode".to_string(), json!("sync")),
            ("top_k".to_string(), json!(5)),
        ]),
        reasoning_effort: None,
    };

    let api_request = to_chat_completion_request(&request, true, &provider_options).unwrap();
    let api_json = serde_json::to_value(api_request).unwrap();

    assert_eq!(api_json["model"], "custom-chat");
    assert_eq!(api_json["stream"], true);
    assert_eq!(api_json["stream_options"], json!({"include_usage": true}));
    assert_eq!(api_json["seed"], 7);
    assert_eq!(api_json["parallel_tool_calls"], false);
    assert_eq!(api_json["user"], "user-42");
    assert_eq!(api_json["service_tier"], "priority");
    assert_eq!(api_json["store"], true);
    assert_eq!(api_json["metadata"], json!({"tenant": "acme"}));
    assert_eq!(api_json["provider_mode"], "sync");
    assert_eq!(api_json["top_k"], 5);
    assert_eq!(api_json["tool_choice"], json!("required"));
    assert_eq!(api_json["tools"][0]["function"]["name"], "search_docs");
    assert_eq!(api_json["tools"][0]["function"]["strict"], true);
    assert_eq!(api_json["response_format"]["type"], "json_schema");
    assert_eq!(api_json["response_format"]["json_schema"]["name"], "answer");
    assert_eq!(
        api_json["messages"][0],
        json!({
            "role": "system",
            "content": "You are a custom provider."
        })
    );
    assert_eq!(api_json["messages"][1]["role"], "user");
    assert_eq!(
        api_json["messages"][1]["content"][0],
        json!({
            "type": "text",
            "text": "Look at this image and search the docs."
        })
    );
    assert_eq!(
        api_json["messages"][1]["content"][1]["image_url"]["url"],
        "https://example.com/cat.png"
    );

    let mut state = SseState::new();
    let mut collector = StreamCollector::new();

    let chunks = [
        json!({
            "id": "resp_custom_1",
            "model": "custom-model",
            "choices": [{
                "index": 0,
                "delta": {"content": "Checking docs... "},
                "finish_reason": null
            }]
        })
        .to_string(),
        json!({
            "id": "resp_custom_1",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": "search_docs",
                            "arguments": "{"
                        }
                    }]
                },
                "finish_reason": null
            }]
        })
        .to_string(),
        json!({
            "id": "resp_custom_1",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": "\"q\":\"rust streaming\"}"
                        }
                    }]
                },
                "finish_reason": null
            }]
        })
        .to_string(),
        json!({
            "id": "resp_custom_1",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls"
            }]
        })
        .to_string(),
        json!({
            "id": "resp_custom_1",
            "model": "custom-model",
            "choices": [],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18
            }
        })
        .to_string(),
        "[DONE]".to_string(),
    ];

    for chunk in chunks {
        for event in process_sse_data(&mut state, &chunk) {
            let event = event.unwrap();
            collector.push(event).unwrap();
        }
    }

    let response = collector.finish().unwrap();
    assert_eq!(response.id.as_deref(), Some("resp_custom_1"));
    assert_eq!(response.model.as_deref(), Some("custom-model"));
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(response.text().as_deref(), Some("Checking docs... "));

    let tool_calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_1");
    assert_eq!(tool_calls[0].name, "search_docs");
    assert_eq!(tool_calls[0].arguments, r#"{"q":"rust streaming"}"#);

    let usage = response.usage.expect("missing usage");
    assert_eq!(usage.input_tokens, Some(11));
    assert_eq!(usage.output_tokens, Some(7));
    assert_eq!(usage.total_tokens, Some(18));
}
