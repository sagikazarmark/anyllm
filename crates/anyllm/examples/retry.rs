use anyllm::prelude::*;
use std::time::Duration;

fn build_provider() -> MockProvider {
    MockProvider::build(|builder| {
        builder
            .error(Error::Timeout("first attempt timed out".to_string()))
            .text("Recovered on retry")
    })
}

fn build_request() -> ChatRequest {
    ChatRequest::new("demo-model").user("Say hello")
}

fn build_retrying_provider(provider: MockProvider) -> RetryingChatProvider<MockProvider> {
    RetryingChatProvider::new(provider).with_policy(RetryPolicy {
        max_attempts: 2,
        base_delay: Duration::ZERO,
        max_delay: Duration::ZERO,
        ..Default::default()
    })
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let provider = build_provider();
    let retrying = build_retrying_provider(provider.clone());
    let request = build_request();

    let response = retrying.chat(&request).await?;
    println!("retry result: {}", response.text_or_empty());
    println!("retry calls: {}", provider.call_count());

    Ok(())
}
