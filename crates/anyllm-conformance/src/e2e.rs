use anyllm::{ChatProvider, ChatRequest, ChatStreamExt, FinishReason, ResponseFormat, Tool};
use serde_json::json;

/// Validate basic chat: non-empty text, Stop finish reason, usage present.
pub async fn basic_chat(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model).user("Say hello in one sentence.");

    let response = provider
        .chat(&request)
        .await
        .expect("basic_chat: chat request failed");

    let text = response.text().expect("basic_chat: response has no text");
    assert!(!text.is_empty(), "basic_chat: response text is empty");

    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "basic_chat: expected Stop finish reason, got {:?}",
        response.finish_reason
    );

    let usage = response
        .usage
        .as_ref()
        .expect("basic_chat: usage is missing");
    assert!(
        usage.input_tokens.is_some_and(|t| t > 0),
        "basic_chat: input_tokens missing or zero"
    );
    assert!(
        usage.output_tokens.is_some_and(|t| t > 0),
        "basic_chat: output_tokens missing or zero"
    );
}

/// Validate streaming: events arrive, collector produces valid response with same structural checks.
pub async fn streaming(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model).user("Say hello in one sentence.");

    let stream = provider
        .chat_stream(&request)
        .await
        .expect("streaming: chat_stream request failed");

    let response = stream
        .collect_response()
        .await
        .expect("streaming: stream collection failed");

    let text = response.text().expect("streaming: response has no text");
    assert!(!text.is_empty(), "streaming: response text is empty");

    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "streaming: expected Stop finish reason, got {:?}",
        response.finish_reason
    );

    let usage = response
        .usage
        .as_ref()
        .expect("streaming: usage is missing");
    assert!(
        usage.input_tokens.is_some_and(|t| t > 0),
        "streaming: input_tokens missing or zero"
    );
    assert!(
        usage.output_tokens.is_some_and(|t| t > 0),
        "streaming: output_tokens missing or zero"
    );
}

/// Validate system prompt: structural checks pass when a system message is included.
pub async fn system_prompt(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model)
        .system("You are a helpful assistant.")
        .user("Say hello in one sentence.");

    let response = provider
        .chat(&request)
        .await
        .expect("system_prompt: chat request failed");

    let text = response
        .text()
        .expect("system_prompt: response has no text");
    assert!(!text.is_empty(), "system_prompt: response text is empty");

    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "system_prompt: expected Stop finish reason, got {:?}",
        response.finish_reason
    );
}

/// Validate multi-turn: a 2-turn conversation is accepted and produces a valid response.
pub async fn multi_turn(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model)
        .user("My name is Alice.")
        .assistant("Hello Alice! How can I help you today?")
        .user("What is my name?");

    let response = provider
        .chat(&request)
        .await
        .expect("multi_turn: chat request failed");

    let text = response.text().expect("multi_turn: response has no text");
    assert!(!text.is_empty(), "multi_turn: response text is empty");

    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "multi_turn: expected Stop finish reason, got {:?}",
        response.finish_reason
    );
}

/// Validate tool calling: model returns tool calls with valid JSON arguments.
pub async fn tool_calling(provider: &impl ChatProvider, model: &str) {
    let tool = Tool::new(
        "get_weather",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"],
            "additionalProperties": false
        }),
    )
    .description("Get the current weather for a city.");

    let request = ChatRequest::new(model)
        .tools(vec![tool])
        .user("What is the weather in San Francisco?");

    let response = provider
        .chat(&request)
        .await
        .expect("tool_calling: chat request failed");

    assert!(
        response.has_tool_calls(),
        "tool_calling: expected tool calls in response"
    );

    for call in response.tool_calls() {
        assert!(
            !call.name.is_empty(),
            "tool_calling: tool call has empty name"
        );
        assert!(
            serde_json::from_str::<serde_json::Value>(call.arguments).is_ok(),
            "tool_calling: tool call arguments are not valid JSON: {}",
            call.arguments
        );
    }

    assert_eq!(
        response.finish_reason,
        Some(FinishReason::ToolCalls),
        "tool_calling: expected ToolCalls finish reason, got {:?}",
        response.finish_reason
    );
}

/// Validate structured output: JSON schema response format produces parseable JSON.
pub async fn structured_output(provider: &impl ChatProvider, model: &str) {
    let schema = json!({
        "type": "object",
        "properties": {
            "greeting": { "type": "string" }
        },
        "required": ["greeting"],
        "additionalProperties": false
    });

    let request = ChatRequest::new(model)
        .response_format(ResponseFormat::JsonSchema {
            name: Some("greeting".into()),
            schema: schema.clone(),
            strict: Some(true),
        })
        .user("Say hello in JSON format with a 'greeting' field.");

    let response = provider
        .chat(&request)
        .await
        .expect("structured_output: chat request failed");

    let text = response
        .text()
        .expect("structured_output: response has no text");

    let parsed: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!("structured_output: response is not valid JSON: {e}\nraw: {text}")
    });

    assert!(
        parsed.get("greeting").is_some(),
        "structured_output: JSON missing 'greeting' field: {parsed}"
    );
}
