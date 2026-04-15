use anyllm::{ChatProvider, ChatRequest};
use anyllm_conformance::{
    FixtureDir, MockHttpResponse, TestHttpServer, assert_response_fixture_eq,
    assert_stream_fixture_eq, load_json_fixture, load_text_fixture,
};
use anyllm_gemini::{ChatRequestOptions, Provider};
use serde_json::json;
use std::path::PathBuf;

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
}

#[tokio::test]
async fn chat_posts_expected_request_and_parses_response() {
    let fixtures = fixtures();
    let raw = load_json_fixture(&fixtures, "response_raw.json");
    let server = TestHttpServer::spawn([MockHttpResponse::json(200, &raw)]).await;

    let provider = Provider::builder()
        .api_key("gemini-test-key")
        .base_url(format!("{}/v1beta/", server.url()))
        .build()
        .unwrap();

    let response = provider
        .chat(
            &ChatRequest::new("gemini-2.5-pro")
                .user("Hello from the transport integration test.")
                .with_option(ChatRequestOptions {
                    candidate_count: Some(1),
                    cached_content: Some("cachedContents/integration-test".into()),
                    ..Default::default()
                }),
        )
        .await
        .unwrap();

    assert_response_fixture_eq(&response, &fixtures, "response_expected.json");

    let requests = server.recorded_requests().await;
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    let body = request.body_json();

    assert_eq!(request.method, "POST");
    assert_eq!(
        request.path,
        "/v1beta/models/gemini-2.5-pro:generateContent"
    );
    assert_eq!(request.header("x-goog-api-key"), Some("gemini-test-key"));
    assert_eq!(request.header("content-type"), Some("application/json"));
    assert_eq!(body["cachedContent"], "cachedContents/integration-test");
    assert_eq!(body["generationConfig"]["candidateCount"], 1);
}

#[tokio::test]
async fn chat_stream_uses_stream_endpoint_and_collects_fixture() {
    let fixtures = fixtures();
    let server = TestHttpServer::spawn([MockHttpResponse::sse(load_text_fixture(
        &fixtures,
        "stream.sse",
    ))])
    .await;

    let provider = Provider::builder()
        .api_key("gemini-test-key")
        .base_url(format!("{}/v1beta/", server.url()))
        .build()
        .unwrap();

    let stream = provider
        .chat_stream(&ChatRequest::new("gemini-2.5-pro").user("Stream this."))
        .await
        .unwrap();

    assert_stream_fixture_eq(
        stream,
        &fixtures,
        "stream_events.json",
        "stream_response_expected.json",
    )
    .await;

    let requests = server.recorded_requests().await;
    assert_eq!(requests.len(), 1);
    let request = &requests[0];

    assert_eq!(
        request.path,
        "/v1beta/models/gemini-2.5-pro:streamGenerateContent"
    );
    assert_eq!(request.query.as_deref(), Some("alt=sse"));
}

#[tokio::test]
async fn chat_maps_non_success_status_to_typed_error() {
    let server = TestHttpServer::spawn([MockHttpResponse::json(
        429,
        &json!({
            "error": {
                "code": 429,
                "message": "Quota exceeded",
                "status": "RESOURCE_EXHAUSTED"
            }
        }),
    )
    .with_header("retry-after", "1.5")])
    .await;

    let provider = Provider::builder()
        .api_key("gemini-test-key")
        .base_url(format!("{}/v1beta/", server.url()))
        .build()
        .unwrap();

    let err = provider
        .chat(&ChatRequest::new("gemini-2.5-pro").user("Trigger an error."))
        .await
        .unwrap_err();

    match err {
        anyllm::Error::RateLimited {
            message,
            retry_after,
            request_id,
        } => {
            assert_eq!(message, "Quota exceeded");
            assert_eq!(retry_after, Some(std::time::Duration::from_secs_f64(1.5)));
            assert_eq!(request_id, None);
        }
        other => panic!("expected RateLimited, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_ignores_invalid_retry_after_header() {
    let server = TestHttpServer::spawn([MockHttpResponse::json(
        429,
        &json!({
            "error": {
                "code": 429,
                "message": "Quota exceeded",
                "status": "RESOURCE_EXHAUSTED"
            }
        }),
    )
    .with_header("retry-after", "-1")])
    .await;

    let provider = Provider::builder()
        .api_key("gemini-test-key")
        .base_url(format!("{}/v1beta/", server.url()))
        .build()
        .unwrap();

    let err = provider
        .chat(&ChatRequest::new("gemini-2.5-pro").user("Trigger an error."))
        .await
        .unwrap_err();

    match err {
        anyllm::Error::RateLimited { retry_after, .. } => {
            assert_eq!(retry_after, None);
        }
        other => panic!("expected RateLimited, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_accepts_success_response_without_candidates() {
    let server = TestHttpServer::spawn([MockHttpResponse::json(
        200,
        &json!({
            "responseId": "resp-empty",
            "modelVersion": "gemini-2.5-pro",
            "candidates": [],
            "usageMetadata": {
                "promptTokenCount": 4,
                "candidatesTokenCount": 0,
                "totalTokenCount": 4
            }
        }),
    )])
    .await;

    let provider = Provider::builder()
        .api_key("gemini-test-key")
        .base_url(format!("{}/v1beta/", server.url()))
        .build()
        .unwrap();

    let response = provider
        .chat(&ChatRequest::new("gemini-2.5-pro").user("Return nothing."))
        .await
        .unwrap();

    assert!(response.content.is_empty());
    assert_eq!(response.id.as_deref(), Some("resp-empty"));
    assert_eq!(response.model.as_deref(), Some("gemini-2.5-pro"));
    let usage = response.usage.expect("expected usage metadata");
    assert_eq!(usage.input_tokens, Some(4));
    assert_eq!(usage.output_tokens, Some(0));
    assert_eq!(usage.total_tokens, Some(4));
}

#[tokio::test]
async fn chat_rejects_blocked_prompt_feedback() {
    let server = TestHttpServer::spawn([MockHttpResponse::json(
        200,
        &json!({
            "responseId": "resp-blocked",
            "modelVersion": "gemini-2.5-pro",
            "promptFeedback": {
                "blockReason": "SAFETY",
                "blockReasonMessage": "Prompt blocked by safety filters"
            },
            "candidates": []
        }),
    )])
    .await;

    let provider = Provider::builder()
        .api_key("gemini-test-key")
        .base_url(format!("{}/v1beta/", server.url()))
        .build()
        .unwrap();

    let err = provider
        .chat(&ChatRequest::new("gemini-2.5-pro").user("Return nothing."))
        .await
        .unwrap_err();

    assert!(matches!(
        err,
        anyllm::Error::ContentFiltered(message) if message == "Prompt blocked by safety filters"
    ));
}

#[tokio::test]
async fn embed_posts_expected_request_and_parses_response() {
    use anyllm::{EmbeddingProvider, EmbeddingRequest};
    use anyllm_conformance::{
        MockHttpResponse, TestHttpServer, assert_embedding_response_fixture_eq, load_json_fixture,
    };

    let fixtures = fixtures();
    let raw = load_json_fixture(&fixtures, "embed_response_raw.json");
    let server = TestHttpServer::spawn([MockHttpResponse::json(200, &raw)]).await;

    let provider = Provider::builder()
        .api_key("test-key")
        .base_url(format!("{}/v1beta", server.url()))
        .build()
        .unwrap();

    let request = EmbeddingRequest::new("text-embedding-004")
        .inputs(["hello world", "gemini embeddings"])
        .dimensions(256);
    let response = provider.embed(&request).await.unwrap();

    assert_embedding_response_fixture_eq(&response, &fixtures, "embed_response_expected.json");

    let recorded = server.recorded_requests().await;
    assert_eq!(recorded.len(), 1);
    let recorded = &recorded[0];
    assert_eq!(recorded.method, "POST");
    assert_eq!(
        recorded.path,
        "/v1beta/models/text-embedding-004:batchEmbedContents"
    );
    assert_eq!(recorded.header("x-goog-api-key"), Some("test-key"));

    let body = recorded.body_json();
    let requests = body["requests"].as_array().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["model"], "models/text-embedding-004");
    assert_eq!(requests[0]["outputDimensionality"], 256);
    assert_eq!(requests[0]["content"]["parts"][0]["text"], "hello world");
}
