use anyllm::prelude::*;

fn build_provider() -> MockEmbeddingProvider {
    MockEmbeddingProvider::new([
        Ok(EmbeddingResponse::new(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
        ])),
        Ok(EmbeddingResponse::new(vec![vec![0.7, 0.8, 0.9]])),
    ])
}

fn build_request() -> EmbeddingRequest {
    EmbeddingRequest::new("demo-embedding-model")
        .inputs(["hello world", "embedding demo"])
        .dimensions(3)
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let provider = build_provider();
    let request = build_request();
    let response = provider.embed(&request).await?;

    for (index, vector) in response.embeddings.iter().enumerate() {
        println!("embedding[{index}] = {vector:?}");
    }

    // Single-input shortcut via the extension trait.
    let single = provider
        .embed_text("demo-embedding-model", "quick one-shot")
        .await?;
    println!("single-input vector: {single:?}");

    println!("embedding calls recorded: {}", provider.requests().len());
    Ok(())
}
