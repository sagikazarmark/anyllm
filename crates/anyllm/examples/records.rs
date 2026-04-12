use anyllm::prelude::*;
use anyllm::{
    ChatRequestRecord, ChatResponseBuilder, ChatResponseRecord, ResponseMetadata,
    ResponseMetadataType,
};
use serde::Serialize;
use serde_json::json;

#[derive(Clone)]
struct DemoRequestOption {
    trace_id: String,
}

#[derive(Clone, Serialize)]
struct DemoResponseMetadata {
    request_id: String,
}

impl ResponseMetadataType for DemoResponseMetadata {
    const KEY: &'static str = "demo";
}

fn build_request() -> ChatRequest {
    ChatRequest::new("demo-model")
        .system("You are a concise assistant.")
        .user("Explain why portable request and response records are useful.")
        .temperature(0.2)
        .max_tokens(120)
        .with_option(DemoRequestOption {
            trace_id: "trace_123".to_string(),
        })
}

fn build_response() -> ChatResponse {
    let mut metadata = ResponseMetadata::new();
    metadata.insert(DemoResponseMetadata {
        request_id: "req_123".to_string(),
    });
    metadata.insert_portable("served_by", json!("mock-provider"));

    ChatResponseBuilder::new()
        .text("Portable records make request/response artifacts easy to inspect, persist, and replay.")
        .model("demo-model")
        .id("resp_123")
        .usage(32, 12)
        .metadata(metadata)
        .build()
}

fn build_provider() -> MockProvider {
    MockProvider::with_response(build_response())
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let request = build_request();
    println!(
        "typed request option before recording: {}",
        request
            .option::<DemoRequestOption>()
            .map(|option| option.trace_id.as_str())
            .unwrap_or("missing")
    );

    let request_record = ChatRequestRecord::from(&request);
    println!(
        "request record:\n{}",
        serde_json::to_string_pretty(&request_record)?
    );

    let replayed_request = request_record.clone().into_chat_request_lossy();
    println!(
        "typed request option after replay: {}",
        replayed_request.option::<DemoRequestOption>().is_some()
    );

    let provider = build_provider();
    let response = provider.chat(&replayed_request).await?;

    let response_record = ChatResponseRecord::from(&response);
    println!(
        "response record:\n{}",
        serde_json::to_string_pretty(&response_record)?
    );

    let replayed_response = response_record.clone().into_chat_response_lossy();
    println!(
        "typed response metadata after replay: {}",
        replayed_response
            .metadata
            .get::<DemoResponseMetadata>()
            .is_some()
    );
    println!("final text: {}", replayed_response.text_or_empty());

    Ok(())
}
