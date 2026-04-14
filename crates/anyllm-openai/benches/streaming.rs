use anyllm::{Result, StreamEvent};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn openai_sse_text() -> &'static str {
    "data: {\"id\":\"chatcmpl-bench\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-bench\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\", world\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-bench\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_bench\",\"type\":\"function\",\"function\":{\"name\":\"search_docs\",\"arguments\":\"{\\\"query\\\":\\\"bench\\\"}\"}}]},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-bench\",\"object\":\"chat.completion.chunk\",\"created\":1710000000,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":24,\"total_tokens\":144}}\n\n\
data: [DONE]\n\n"
}

fn bench_openai_streaming(c: &mut Criterion) {
    let text = openai_sse_text();
    c.bench_function("openai_stream_event_translation", |b| {
        b.iter(|| {
            futures_executor::block_on(async {
                use futures_util::StreamExt;
                let mut stream = anyllm_openai::conformance_stream_from_sse_text(black_box(text));
                let mut events = Vec::<Result<StreamEvent>>::new();
                while let Some(event) = stream.next().await {
                    events.push(event);
                }
                black_box(events)
            })
        })
    });
}

criterion_group!(benches, bench_openai_streaming);
criterion_main!(benches);
