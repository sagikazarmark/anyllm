use anyllm::prelude::*;
use std::time::Duration;

fn build_request() -> ChatRequest {
    ChatRequest::new("demo-model").user("Say hello")
}

fn build_primary() -> MockProvider {
    MockProvider::build(|builder| {
        builder
            .error(Error::Overloaded {
                message: "primary overloaded".to_string(),
                retry_after: Some(Duration::from_millis(25)),
                request_id: Some("req_primary_1".to_string()),
            })
            .provider_name("primary-mock")
    })
}

fn build_fallback() -> MockProvider {
    MockProvider::build(|builder| {
        builder
            .text("Served by fallback provider")
            .provider_name("fallback-mock")
    })
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let primary = build_primary();
    let fallback_provider = build_fallback();
    let fallback = FallbackChatProvider::new(primary.clone(), fallback_provider.clone());
    let request = build_request();

    let response = fallback.chat(&request).await?;
    println!("fallback result: {}", response.text_or_empty());
    println!("primary calls: {}", primary.call_count());
    println!("fallback calls: {}", fallback_provider.call_count());

    Ok(())
}
