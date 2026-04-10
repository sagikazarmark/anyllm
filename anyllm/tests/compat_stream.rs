use anyllm::{
    ContentBlock, ExtraMap, FinishReason, StreamBlockType, StreamCollector, StreamEvent, Usage,
    UsageMetadataMode,
};
use serde_json::json;

#[test]
fn stream_completeness_serialization_is_stable() {
    let completeness = anyllm::StreamCompleteness {
        saw_response_start: true,
        saw_response_stop: false,
        has_open_blocks: true,
        dropped_incomplete_tool_calls: false,
    };

    let value = serde_json::to_value(completeness).unwrap();
    assert_eq!(
        value,
        json!({
            "saw_response_start": true,
            "saw_response_stop": false,
            "has_open_blocks": true,
            "dropped_incomplete_tool_calls": false
        })
    );
}

#[test]
fn stream_event_serialization_is_stable() {
    let response_start = serde_json::to_value(StreamEvent::ResponseStart {
        id: Some("resp_1".into()),
        model: Some("model-a".into()),
    })
    .unwrap();
    assert_eq!(
        response_start,
        json!({
            "event": "response_start",
            "id": "resp_1",
            "model": "model-a"
        })
    );

    let block_start = serde_json::to_value(StreamEvent::BlockStart {
        index: 2,
        block_type: StreamBlockType::ToolCall,
        id: Some("call_1".into()),
        name: Some("search".into()),
        type_name: None,
        data: None,
    })
    .unwrap();
    assert_eq!(
        block_start,
        json!({
            "event": "block_start",
            "index": 2,
            "block_type": "tool_call",
            "id": "call_1",
            "name": "search"
        })
    );

    let reasoning_delta = serde_json::to_value(StreamEvent::ReasoningDelta {
        index: 1,
        text: "thinking".into(),
        signature: Some("sig_1".into()),
    })
    .unwrap();
    assert_eq!(
        reasoning_delta,
        json!({
            "event": "reasoning_delta",
            "index": 1,
            "text": "thinking",
            "signature": "sig_1"
        })
    );

    let metadata = serde_json::to_value(StreamEvent::ResponseMetadata {
        finish_reason: Some(FinishReason::ToolCalls),
        usage: Some(Usage::new().input_tokens(10).output_tokens(5)),
        usage_mode: UsageMetadataMode::Snapshot,
        id: Some("resp_1".into()),
        model: Some("model-a".into()),
        metadata: ExtraMap::new(),
    })
    .unwrap();
    assert_eq!(
        metadata,
        json!({
            "event": "response_metadata",
            "finish_reason": "tool_calls",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            },
            "id": "resp_1",
            "model": "model-a"
        })
    );

    let stop = serde_json::to_value(StreamEvent::ResponseStop).unwrap();
    assert_eq!(stop, json!({"event": "response_stop"}));
}

#[test]
fn stream_collector_reconstructs_ordered_response_contract() {
    let mut collector = StreamCollector::new();

    let events = [
        StreamEvent::ResponseStart {
            id: Some("resp_1".into()),
            model: Some("model-a".into()),
        },
        StreamEvent::BlockStart {
            index: 1,
            block_type: StreamBlockType::Reasoning,
            id: None,
            name: None,
            type_name: None,
            data: None,
        },
        StreamEvent::ReasoningDelta {
            index: 1,
            text: "let me think".into(),
            signature: Some("sig_1".into()),
        },
        StreamEvent::BlockStop { index: 1 },
        StreamEvent::BlockStart {
            index: 0,
            block_type: StreamBlockType::Text,
            id: None,
            name: None,
            type_name: None,
            data: None,
        },
        StreamEvent::TextDelta {
            index: 0,
            text: "answer".into(),
        },
        StreamEvent::BlockStop { index: 0 },
        StreamEvent::BlockStart {
            index: 2,
            block_type: StreamBlockType::ToolCall,
            id: Some("call_1".into()),
            name: Some("search".into()),
            type_name: None,
            data: None,
        },
        StreamEvent::ToolCallDelta {
            index: 2,
            arguments: r#"{"q":"rust"}"#.into(),
        },
        StreamEvent::BlockStop { index: 2 },
        StreamEvent::BlockStart {
            index: 3,
            block_type: StreamBlockType::Other,
            id: None,
            name: None,
            type_name: Some("citation".into()),
            data: Some(serde_json::Map::from_iter([(
                "url".into(),
                json!("https://example.com"),
            )])),
        },
        StreamEvent::BlockStop { index: 3 },
        StreamEvent::ResponseMetadata {
            finish_reason: None,
            usage: Some(Usage::new().input_tokens(10).output_tokens(3)),
            usage_mode: UsageMetadataMode::Delta,
            id: None,
            model: None,
            metadata: ExtraMap::new(),
        },
        StreamEvent::ResponseMetadata {
            finish_reason: Some(FinishReason::ToolCalls),
            usage: Some(Usage::new().output_tokens(2).total_tokens(12)),
            usage_mode: UsageMetadataMode::Delta,
            id: Some("resp_1b".into()),
            model: Some("model-b".into()),
            metadata: ExtraMap::new(),
        },
        StreamEvent::ResponseStop,
    ];

    for event in &events {
        collector.push(event.clone()).unwrap();
    }

    let response = collector.finish().unwrap();
    assert_eq!(response.id.as_deref(), Some("resp_1b"));
    assert_eq!(response.model.as_deref(), Some("model-b"));
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));

    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.output_tokens, Some(5));
    assert_eq!(usage.total_tokens, Some(12));

    assert_eq!(
        response.content,
        vec![
            ContentBlock::Text {
                text: "answer".into(),
            },
            ContentBlock::Reasoning {
                text: "let me think".into(),
                signature: Some("sig_1".into()),
            },
            ContentBlock::ToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: r#"{"q":"rust"}"#.into(),
            },
            ContentBlock::Other {
                type_name: "citation".into(),
                data: serde_json::Map::from_iter([("url".into(), json!("https://example.com"),)]),
            },
        ]
    );
}

#[test]
fn stream_collector_finish_is_strict_about_missing_block_stop() {
    let mut collector = StreamCollector::new();

    let events = [
        StreamEvent::ResponseStart {
            id: Some("resp_1".into()),
            model: Some("model-a".into()),
        },
        StreamEvent::BlockStart {
            index: 0,
            block_type: StreamBlockType::Text,
            id: None,
            name: None,
            type_name: None,
            data: None,
        },
        StreamEvent::TextDelta {
            index: 0,
            text: "answer".into(),
        },
        StreamEvent::ResponseStop,
    ];

    for event in &events {
        collector.push(event.clone()).unwrap();
    }

    let err = collector.finish().unwrap_err();
    assert!(matches!(
        err,
        anyllm::Error::Stream(message)
            if message
                == "stream incomplete: start=true, stop=true, open_blocks=true, dropped_incomplete_tool_calls=false, terminal_error=none"
    ));
}

#[test]
fn stream_collector_finish_partial_preserves_incomplete_non_tool_content() {
    let mut collector = StreamCollector::new();

    let events = [
        StreamEvent::ResponseStart {
            id: Some("resp_1".into()),
            model: Some("model-a".into()),
        },
        StreamEvent::BlockStart {
            index: 0,
            block_type: StreamBlockType::Text,
            id: None,
            name: None,
            type_name: None,
            data: None,
        },
        StreamEvent::TextDelta {
            index: 0,
            text: "answer".into(),
        },
        StreamEvent::ResponseStop,
    ];

    for event in &events {
        collector.push(event.clone()).unwrap();
    }

    let collected = collector.finish_partial().unwrap();
    assert_eq!(collected.response.text(), Some("answer".into()));
    assert!(collected.completeness.saw_response_start);
    assert!(collected.completeness.saw_response_stop);
    assert!(collected.completeness.has_open_blocks);
    assert!(!collected.completeness.dropped_incomplete_tool_calls);
    assert!(!collected.completeness.is_complete());
}
