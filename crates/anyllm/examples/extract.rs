use anyllm::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Review {
    title: String,
    rating: u8,
}

fn build_provider() -> MockProvider {
    MockProvider::with_text(r#"{"title":"Dune","rating":9}"#)
        .with_supported_chat_capabilities([ChatCapability::StructuredOutput])
}

fn build_request() -> ChatRequest {
    ChatRequest::new("demo-model")
        .system("Extract a review as structured data that matches the requested schema.")
        .user("Dune is atmospheric, visually striking, and deserves a 9 out of 10.")
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let provider = build_provider();
    let request = build_request();

    let extracted: Extracted<Review> = provider.extract(&request).await?;
    println!("title: {}", extracted.value.title);
    println!("rating: {}", extracted.value.rating);
    println!("passes: {}", extracted.metadata.passes);
    println!("repaired: {}", extracted.metadata.repaired);
    println!("raw text: {}", extracted.response.text_or_empty());

    Ok(())
}
