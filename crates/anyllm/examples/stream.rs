use anyllm::prelude::*;

fn build_provider() -> MockStreamingProvider {
    MockStreamingProvider::build(|builder| {
        builder.text("Deterministic streamed hello from anyllm.")
    })
}

fn build_request() -> ChatRequest {
    ChatRequest::new("demo-model").user("Say hello")
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let provider = build_provider();
    let request = build_request();
    let mut stream = provider.chat_stream(&request).await?;
    let mut collector = StreamCollector::new();

    while let Some(event) = stream.next().await {
        let event = event?;
        println!("stream event: {event:?}");
        collector.push(event)?;
    }

    let response = collector.finish()?;
    println!("final text: {}", response.text_or_empty());
    println!("stream calls recorded: {}", provider.call_count());
    Ok(())
}
