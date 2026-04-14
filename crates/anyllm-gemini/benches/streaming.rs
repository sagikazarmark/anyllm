use anyllm::{Result, StreamEvent};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn gemini_sse_text() -> String {
    let chunks = [
        r#"{"responseId":"resp_bench","modelVersion":"gemini-2.5-pro","candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}"#,
        r#"{"responseId":"resp_bench","modelVersion":"gemini-2.5-pro","candidates":[{"content":{"parts":[{"text":"Hello, world"},{"text":" and reasoning","thought":true}]}}]}"#,
        r#"{"responseId":"resp_bench","modelVersion":"gemini-2.5-pro","candidates":[{"content":{"parts":[{"functionCall":{"name":"search_docs","args":{"query":"bench","limit":3}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":120,"candidatesTokenCount":24,"totalTokenCount":144}}"#,
    ];
    chunks
        .into_iter()
        .map(|chunk| format!("data: {chunk}\n\n"))
        .collect()
}

fn parse_stream(text: &str) -> Vec<Result<StreamEvent>> {
    futures_executor::block_on(async move {
        use futures_util::StreamExt;
        let mut stream = anyllm_gemini::conformance_stream_from_sse_text(text);
        let mut out = Vec::new();
        while let Some(event) = stream.next().await {
            out.push(event);
        }
        out
    })
}

fn bench_gemini_streaming(c: &mut Criterion) {
    let text = gemini_sse_text();
    c.bench_function("gemini_stream_event_translation", |b| {
        b.iter(|| black_box(parse_stream(black_box(&text))))
    });
}

criterion_group!(benches, bench_gemini_streaming);
criterion_main!(benches);
