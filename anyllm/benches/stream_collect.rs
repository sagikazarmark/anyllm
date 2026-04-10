use anyllm::{
    ExtraMap, FinishReason, StreamBlockType, StreamCollector, StreamEvent, Usage, UsageMetadataMode,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn sample_events() -> Vec<StreamEvent> {
    let mut events = vec![StreamEvent::ResponseStart {
        id: Some("resp_bench".into()),
        model: Some("bench-model".into()),
    }];

    for index in 0..24 {
        events.push(StreamEvent::BlockStart {
            index,
            block_type: if index % 3 == 0 {
                StreamBlockType::ToolCall
            } else if index % 3 == 1 {
                StreamBlockType::Text
            } else {
                StreamBlockType::Reasoning
            },
            id: (index % 3 == 0).then(|| format!("call_{index}")),
            name: (index % 3 == 0).then(|| format!("tool_{index}")),
            type_name: None,
            data: None,
        });

        match index % 3 {
            0 => events.push(StreamEvent::ToolCallDelta {
                index,
                arguments: format!(r#"{{"value":{index},"payload":"{}"}}"#, "x".repeat(96)),
            }),
            1 => events.push(StreamEvent::TextDelta {
                index,
                text: "Lorem ipsum dolor sit amet. ".repeat(6),
            }),
            _ => events.push(StreamEvent::ReasoningDelta {
                index,
                text: "Deliberating carefully. ".repeat(5),
                signature: (index == 2).then(|| "sig_bench".to_string()),
            }),
        }

        events.push(StreamEvent::BlockStop { index });
    }

    events.push(StreamEvent::ResponseMetadata {
        finish_reason: Some(FinishReason::Stop),
        usage: Some(
            Usage::new()
                .input_tokens(1024)
                .output_tokens(256)
                .total_tokens(1280),
        ),
        usage_mode: UsageMetadataMode::Snapshot,
        id: Some("resp_bench".into()),
        model: Some("bench-model".into()),
        metadata: ExtraMap::new(),
    });
    events.push(StreamEvent::ResponseStop);
    events
}

fn bench_stream_collect(c: &mut Criterion) {
    let events = sample_events();
    c.bench_function("stream_collector_collect_response", |b| {
        b.iter(|| {
            let mut collector = StreamCollector::new();
            for event in black_box(&events) {
                collector.push(event.clone()).unwrap();
            }
            black_box(collector.finish().unwrap())
        })
    });
}

criterion_group!(benches, bench_stream_collect);
criterion_main!(benches);
