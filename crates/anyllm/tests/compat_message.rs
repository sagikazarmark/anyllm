use anyllm::{ContentBlock, ContentPart, ExtraMap, ImageSource, Message};
use serde_json::json;

#[test]
fn system_message_serializes_with_extensions_field() {
    let message = Message::system("You are helpful")
        .with_extension("cache_control", json!({"type": "ephemeral"}));

    let value = serde_json::to_value(&message).unwrap();
    assert_eq!(
        value,
        json!({
            "role": "system",
            "content": "You are helpful",
            "extensions": {
                "cache_control": {"type": "ephemeral"}
            }
        })
    );
}

#[test]
fn user_multimodal_message_serializes_with_name_and_parts() {
    let message = Message::User {
        content: anyllm::UserContent::Parts(vec![
            ContentPart::Text {
                text: "look at this".into(),
            },
            ContentPart::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png".into(),
                },
                detail: Some("high".into()),
            },
        ]),
        name: Some("alice".into()),
        extensions: Some(ExtraMap::from_iter([("trace_id".into(), json!(123))])),
    };

    let value = serde_json::to_value(&message).unwrap();
    assert_eq!(
        value,
        json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/cat.png"},
                    "detail": "high"
                }
            ],
            "name": "alice",
            "extensions": {
                "trace_id": 123
            }
        })
    );
}

#[test]
fn assistant_message_serializes_content_blocks_and_extensions() {
    let message = Message::Assistant {
        content: vec![
            ContentBlock::Reasoning {
                text: "thinking".into(),
                signature: Some("sig_1".into()),
            },
            ContentBlock::Text {
                text: "done".into(),
            },
        ],
        name: Some("assistant-1".into()),
        extensions: Some(ExtraMap::from_iter([("provider_hint".into(), json!(true))])),
    };

    let value = serde_json::to_value(&message).unwrap();
    assert_eq!(
        value,
        json!({
            "role": "assistant",
            "content": [
                {"type": "reasoning", "text": "thinking", "signature": "sig_1"},
                {"type": "text", "text": "done"}
            ],
            "name": "assistant-1",
            "extensions": {
                "provider_hint": true
            }
        })
    );
}

#[test]
fn tool_message_serializes_is_error_only_when_present() {
    let message = Message::tool_error("call_1", "search", "failed");

    let value = serde_json::to_value(&message).unwrap();
    assert_eq!(
        value,
        json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "search",
            "content": "failed",
            "is_error": true,
        })
    );
}

#[test]
fn message_deserialize_accepts_legacy_extra_field() {
    let value = json!({
        "role": "system",
        "content": "hello",
        "extra": {
            "cache_control": {"type": "ephemeral"}
        }
    });

    let message: Message = serde_json::from_value(value).unwrap();
    assert_eq!(
        serde_json::to_value(&message).unwrap(),
        json!({
            "role": "system",
            "content": "hello",
            "extensions": {
                "cache_control": {"type": "ephemeral"}
            }
        })
    );
}

#[test]
fn message_deserialize_promotes_unknown_fields_into_extensions() {
    let value = json!({
        "role": "system",
        "content": "hello",
        "cache_control": {"type": "ephemeral"},
        "priority": 5
    });

    let message: Message = serde_json::from_value(value).unwrap();

    match message {
        Message::System { extensions, .. } => {
            assert_eq!(
                extensions,
                Some(ExtraMap::from_iter([
                    ("cache_control".into(), json!({"type": "ephemeral"})),
                    ("priority".into(), json!(5)),
                ]))
            );
        }
        other => panic!("expected system message, got {other:?}"),
    }
}
