use anyllm::{
    ChatResponse, ContentBlock, FinishReason, ReasoningConfig, ReasoningEffort, ResponseFormat,
    ResponseMetadata, Usage,
};
use serde_json::json;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct DemoMetadata {
    request_id: String,
}

impl anyllm::ResponseMetadataType for DemoMetadata {
    const KEY: &'static str = "demo";
}

#[test]
fn reasoning_config_serialization_is_stable() {
    let config = ReasoningConfig {
        enabled: true,
        budget_tokens: Some(1024),
        effort: Some(ReasoningEffort::High),
    };

    assert_eq!(
        serde_json::to_value(config).unwrap(),
        json!({
            "enabled": true,
            "budget_tokens": 1024,
            "effort": "high"
        })
    );

    let minimal = ReasoningConfig {
        enabled: false,
        budget_tokens: None,
        effort: None,
    };
    assert_eq!(
        serde_json::to_value(minimal).unwrap(),
        json!({"enabled": false})
    );
}

#[test]
fn response_format_serialization_is_stable() {
    assert_eq!(
        serde_json::to_value(ResponseFormat::Text).unwrap(),
        json!("text")
    );
    assert_eq!(
        serde_json::to_value(ResponseFormat::Json).unwrap(),
        json!("json")
    );
    assert_eq!(
        serde_json::to_value(ResponseFormat::JsonSchema {
            name: Some("schema_name".into()),
            schema: json!({"type": "object", "properties": {"x": {"type": "string"}}}),
            strict: Some(true),
        })
        .unwrap(),
        json!({
            "json_schema": {
                "name": "schema_name",
                "schema": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}}
                },
                "strict": true
            }
        })
    );
}

#[test]
fn usage_serialization_is_stable() {
    let usage = Usage::new()
        .input_tokens(10)
        .output_tokens(5)
        .total_tokens(15)
        .cached_input_tokens(3)
        .cache_creation_input_tokens(2)
        .reasoning_tokens(1);

    assert_eq!(
        serde_json::to_value(usage).unwrap(),
        json!({
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "cached_input_tokens": 3,
            "cache_creation_input_tokens": 2,
            "reasoning_tokens": 1
        })
    );
}

#[test]
fn chat_response_log_value_is_stable() {
    let mut metadata = ResponseMetadata::new();
    metadata.insert(DemoMetadata {
        request_id: "req_123".into(),
    });

    let response = ChatResponse::new(vec![
        ContentBlock::Reasoning {
            text: "thinking".into(),
            signature: Some("sig_1".into()),
        },
        ContentBlock::Text {
            text: "done".into(),
        },
    ])
    .finish_reason(FinishReason::Stop)
    .usage(
        Usage::new()
            .input_tokens(10)
            .output_tokens(5)
            .total_tokens(15),
    )
    .model("gpt-4o")
    .id("resp_1")
    .metadata(metadata);

    assert_eq!(
        response.to_log_value(),
        json!({
            "content": [
                {"type": "reasoning", "text": "thinking", "signature": "sig_1"},
                {"type": "text", "text": "done"}
            ],
            "finish_reason": "stop",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15
            },
            "model": "gpt-4o",
            "id": "resp_1",
            "metadata": {
                "demo": {
                    "request_id": "req_123"
                }
            }
        })
    );
}
