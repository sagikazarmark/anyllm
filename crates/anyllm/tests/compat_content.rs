use anyllm::{ContentBlock, ContentPart, FinishReason, ImageSource};
use serde_json::json;

#[test]
fn content_part_serialization_is_flat_and_tagged() {
    let text = serde_json::to_value(ContentPart::Text {
        text: "hello".into(),
    })
    .unwrap();
    assert_eq!(text, json!({"type": "text", "text": "hello"}));

    let image = serde_json::to_value(ContentPart::Image {
        source: ImageSource::Base64 {
            media_type: "image/png".into(),
            data: "abc123".into(),
        },
        detail: Some("low".into()),
    })
    .unwrap();
    assert_eq!(
        image,
        json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc123"
            },
            "detail": "low"
        })
    );
}

#[test]
fn content_part_other_round_trips_unknown_type_name() {
    let input = json!({
        "type": "audio",
        "format": "wav",
        "url": "https://example.com/a.wav"
    });

    let part: ContentPart = serde_json::from_value(input.clone()).unwrap();
    let output = serde_json::to_value(part).unwrap();
    assert_eq!(output, input);
}

#[test]
fn content_block_serialization_is_flat_and_tagged() {
    let text = serde_json::to_value(ContentBlock::Text {
        text: "hello".into(),
    })
    .unwrap();
    assert_eq!(text, json!({"type": "text", "text": "hello"}));

    let tool_call = serde_json::to_value(ContentBlock::ToolCall {
        id: "call_1".into(),
        name: "search".into(),
        arguments: r#"{"q":"rust"}"#.into(),
    })
    .unwrap();
    assert_eq!(
        tool_call,
        json!({
            "type": "tool_call",
            "id": "call_1",
            "name": "search",
            "arguments": r#"{"q":"rust"}"#
        })
    );

    let reasoning = serde_json::to_value(ContentBlock::Reasoning {
        text: "thinking".into(),
        signature: None,
    })
    .unwrap();
    assert_eq!(reasoning, json!({"type": "reasoning", "text": "thinking"}));
}

#[test]
fn content_block_other_round_trips_unknown_type_name() {
    let input = json!({
        "type": "citation",
        "source": "https://example.com",
        "title": "Example"
    });

    let block: ContentBlock = serde_json::from_value(input.clone()).unwrap();
    let output = serde_json::to_value(block).unwrap();
    assert_eq!(output, input);
}

#[test]
fn finish_reason_serialization_uses_stable_strings() {
    assert_eq!(
        serde_json::to_value(FinishReason::Stop).unwrap(),
        json!("stop")
    );
    assert_eq!(
        serde_json::to_value(FinishReason::Length).unwrap(),
        json!("length")
    );
    assert_eq!(
        serde_json::to_value(FinishReason::ToolCalls).unwrap(),
        json!("tool_calls")
    );
    assert_eq!(
        serde_json::to_value(FinishReason::ContentFilter).unwrap(),
        json!("content_filter")
    );
    assert_eq!(
        serde_json::to_value(FinishReason::Other("provider_stop".into())).unwrap(),
        json!("provider_stop")
    );
}

#[test]
fn finish_reason_unknown_string_deserializes_to_other() {
    let finish: FinishReason = serde_json::from_value(json!("custom_stop")).unwrap();
    assert_eq!(finish, FinishReason::Other("custom_stop".into()));
}
