use anyllm::{CapabilitySupport, EmbeddingCapability, EmbeddingProvider, EmbeddingRequest};
use anyllm_conformance::{MockHttpResponse, TestHttpServer};
use anyllm_openai_compat::providers::Cloudflare;

#[tokio::test]
async fn cloudflare_openai_compat_embed_posts_expected_request() {
    let response_body = serde_json::json!({
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0},
            {"embedding": [0.4, 0.5, 0.6], "index": 1}
        ],
        "model": "@cf/baai/bge-base-en-v1.5",
        "usage": {"prompt_tokens": 4, "total_tokens": 4}
    });

    let server = TestHttpServer::spawn([MockHttpResponse::json(200, &response_body)]).await;

    let provider = Cloudflare::builder("account-123", "token-abc")
        .unwrap()
        .base_url(format!("{}/v1", server.url()))
        .build()
        .unwrap();

    let request = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5").inputs(["a", "b"]);

    let response = provider.embed(&request).await.unwrap();
    assert_eq!(response.model.as_deref(), Some("@cf/baai/bge-base-en-v1.5"));
    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(response.embeddings[0].len(), 3);
    assert_eq!(response.embeddings[1].len(), 3);

    let recorded = server.recorded_requests().await;
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].path, "/v1/embeddings");
    assert_eq!(
        recorded[0].header("authorization"),
        Some("Bearer token-abc")
    );

    assert_eq!(
        provider
            .embedding_capability("@cf/baai/bge-base-en-v1.5", EmbeddingCapability::BatchInput,),
        CapabilitySupport::Supported
    );
    assert_eq!(
        provider.embedding_capability(
            "@cf/baai/bge-base-en-v1.5",
            EmbeddingCapability::OutputDimensions,
        ),
        CapabilitySupport::Unknown
    );
}
