use anyllm::prelude::*;

fn build_provider() -> MockProvider {
    MockProvider::build(|builder| builder.text("Deterministic hello from anyllm."))
}

fn build_request() -> ChatRequest {
    ChatRequest::new("demo-model").user("Say hello")
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let provider = build_provider();
    let request = build_request();
    let response = provider.chat(&request).await?;

    println!("chat text: {}", response.text_or_empty());
    println!("chat calls recorded: {}", provider.call_count());
    Ok(())
}
