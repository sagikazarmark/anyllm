#![cfg(feature = "mock")]

use anyllm::prelude::*;

#[test]
fn prelude_exposes_embedding_types() {
    let request = EmbeddingRequest::new("model").input("hello").dimensions(32);
    assert_eq!(request.model, "model");
    assert_eq!(request.dimensions, Some(32));

    let response = EmbeddingResponse::new(vec![vec![0.0, 1.0]]).model("m");
    assert_eq!(response.embeddings.len(), 1);
}

#[tokio::test]
async fn prelude_exposes_mock_embedding_provider() {
    let provider = MockEmbeddingProvider::new([
        Ok(EmbeddingResponse::new(vec![vec![0.5]])),
        Ok(EmbeddingResponse::new(vec![vec![0.7]])),
    ]);
    let response = provider
        .embed(&EmbeddingRequest::new("m").input("x"))
        .await
        .unwrap();
    assert_eq!(response.embeddings, vec![vec![0.5]]);

    let vector = provider
        .with_provider_name("prelude-test")
        .embed_text("m", "y")
        .await
        .unwrap();
    assert_eq!(vector, vec![0.7]);
}
