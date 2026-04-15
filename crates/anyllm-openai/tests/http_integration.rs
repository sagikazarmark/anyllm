use anyllm::{ChatProvider, ChatRequest, ContentPart, ImageSource, Message};
use anyllm_conformance::{
    FixtureDir, MockHttpResponse, TestHttpServer, assert_partial_stream_fixture_eq,
    assert_response_fixture_eq, assert_stream_finish_error_fixture_eq, assert_stream_fixture_eq,
    load_json_fixture, load_text_fixture,
};
use anyllm_openai::{ChatRequestOptions, ChatResponseMetadata, Provider};
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
        .api_key("sk-test")
        .base_url(format!("{}/v1/", server.url()))
        .organization("org-test")
        .project("proj-test")
        .build()
        .unwrap();

    let response = provider
        .chat(
            &ChatRequest::new("gpt-4o")
                .user("Hello from the transport integration test.")
                .with_option(ChatRequestOptions {
                    user: Some("user-123".into()),
                    ..Default::default()
                }),
        )
        .await
        .unwrap();

    assert_response_fixture_eq(&response, &fixtures, "response_expected.json");
    assert_eq!(
        response.metadata.get::<ChatResponseMetadata>(),
        Some(&ChatResponseMetadata {
            system_fingerprint: Some("fp_123".into()),
        })
    );

    let requests = server.recorded_requests().await;
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    let body = request.body_json();

    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(request.header("authorization"), Some("Bearer sk-test"));
    assert_eq!(request.header("openai-organization"), Some("org-test"));
    assert_eq!(request.header("openai-project"), Some("proj-test"));
    assert_eq!(request.header("content-type"), Some("application/json"));
    assert_eq!(body["model"], "gpt-4o");
    assert_eq!(body["user"], "user-123");
}

#[tokio::test]
async fn chat_stream_uses_stream_mode_and_collects_fixture() {
    let fixtures = fixtures();
    let server = TestHttpServer::spawn([MockHttpResponse::sse(load_text_fixture(
        &fixtures,
        "stream.sse",
    ))])
    .await;

    let provider = Provider::builder()
        .api_key("sk-test")
        .base_url(format!("{}/v1/", server.url()))
        .build()
        .unwrap();

    let stream = provider
        .chat_stream(&ChatRequest::new("gpt-4o").user("Stream this response."))
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
    let body = request.body_json();

    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(body["stream"], true);
}

#[tokio::test]
async fn chat_stream_clean_eof_reports_incomplete_stream() {
    let fixtures = fixtures();
    let truncated = load_text_fixture(&fixtures, "stream_truncated.sse");
    let server = TestHttpServer::spawn([
        MockHttpResponse::sse(truncated.clone()),
        MockHttpResponse::sse(truncated),
    ])
    .await;

    let provider = Provider::builder()
        .api_key("sk-test")
        .base_url(format!("{}/v1/", server.url()))
        .build()
        .unwrap();

    let stream = provider
        .chat_stream(&ChatRequest::new("gpt-4o").user("Stream this response."))
        .await
        .unwrap();

    assert_stream_finish_error_fixture_eq(
        stream,
        &fixtures,
        "stream_truncated_events.json",
        "stream_truncated_error.json",
    )
    .await;

    let stream = provider
        .chat_stream(&ChatRequest::new("gpt-4o").user("Stream this response."))
        .await
        .unwrap();

    assert_partial_stream_fixture_eq(
        stream,
        &fixtures,
        "stream_truncated_events.json",
        "stream_truncated_response_expected.json",
        "stream_truncated_completeness.json",
    )
    .await;
}

#[tokio::test]
async fn chat_maps_non_success_status_to_typed_error() {
    let server = TestHttpServer::spawn([MockHttpResponse::json(
        429,
        &json!({
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit reached"
            }
        }),
    )
    .with_header("x-request-id", "req-openai-429")
    .with_header("retry-after", "2.5")])
    .await;

    let provider = Provider::builder()
        .api_key("sk-test")
        .base_url(format!("{}/v1/", server.url()))
        .build()
        .unwrap();

    let err = provider
        .chat(&ChatRequest::new("gpt-4o").user("Trigger an error."))
        .await
        .unwrap_err();

    match err {
        anyllm::Error::RateLimited {
            message,
            retry_after,
            request_id,
        } => {
            assert_eq!(message, "Rate limit reached");
            assert_eq!(retry_after, Some(std::time::Duration::from_secs_f64(2.5)));
            assert_eq!(request_id.as_deref(), Some("req-openai-429"));
        }
        other => panic!("expected RateLimited, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_posts_openai_vision_request_shape() {
    let fixtures = fixtures();
    let raw = load_json_fixture(&fixtures, "response_raw.json");
    let server = TestHttpServer::spawn([MockHttpResponse::json(200, &raw)]).await;

    let provider = Provider::builder()
        .api_key("sk-test")
        .base_url(format!("{}/v1/", server.url()))
        .build()
        .unwrap();

    provider
        .chat(
            &ChatRequest::new("gpt-4o").message(Message::user_multimodal(vec![
                ContentPart::text("Describe this image."),
                ContentPart::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/cat.png".into(),
                    },
                    detail: Some("high".into()),
                },
            ])),
        )
        .await
        .unwrap();

    let requests = server.recorded_requests().await;
    assert_eq!(requests.len(), 1);
    let body = requests[0].body_json();
    let parts = body["messages"][0]["content"]
        .as_array()
        .expect("expected multimodal content array");

    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["text"], "Describe this image.");
    assert_eq!(parts[1]["type"], "image_url");
    assert_eq!(parts[1]["image_url"]["url"], "https://example.com/cat.png");
    assert_eq!(parts[1]["image_url"]["detail"], "high");
}

#[tokio::test]
async fn embed_posts_expected_request_and_parses_response() {
    use anyllm::{EmbeddingProvider, EmbeddingRequest};
    use anyllm_conformance::assert_embedding_response_fixture_eq;

    let fixtures = fixtures();
    let raw = load_json_fixture(&fixtures, "embed_response_raw.json");
    let server = TestHttpServer::spawn([MockHttpResponse::json(200, &raw)]).await;

    let provider = Provider::builder()
        .api_key("sk-test")
        .base_url(format!("{}/v1/", server.url()))
        .organization("org-test")
        .project("proj-test")
        .build()
        .unwrap();

    let request = EmbeddingRequest::new("text-embedding-3-small")
        .inputs(["hello world", "embedding fixtures"])
        .dimensions(32);
    let response = provider.embed(&request).await.unwrap();

    assert_embedding_response_fixture_eq(&response, &fixtures, "embed_response_expected.json");

    let recorded = server.recorded_requests().await;
    assert_eq!(recorded.len(), 1);
    let recorded = &recorded[0];
    assert_eq!(recorded.method, "POST");
    assert_eq!(recorded.path, "/v1/embeddings");
    assert_eq!(recorded.header("authorization"), Some("Bearer sk-test"));
    assert_eq!(recorded.header("openai-organization"), Some("org-test"));
    assert_eq!(recorded.header("openai-project"), Some("proj-test"));

    let body = recorded.body_json();
    assert_eq!(body["model"], "text-embedding-3-small");
    assert_eq!(
        body["input"],
        serde_json::json!(["hello world", "embedding fixtures"])
    );
    assert_eq!(body["dimensions"], 32);
}

#[tokio::test]
async fn embed_maps_401_to_auth_error() {
    use anyllm::{EmbeddingProvider, EmbeddingRequest, Error};

    let server = TestHttpServer::spawn([MockHttpResponse::json(
        401,
        &serde_json::json!({
            "error": {
                "type": "invalid_request_error",
                "message": "Incorrect API key provided"
            }
        }),
    )])
    .await;

    let provider = Provider::builder()
        .api_key("sk-test")
        .base_url(format!("{}/v1/", server.url()))
        .build()
        .unwrap();

    let err = provider
        .embed(&EmbeddingRequest::new("text-embedding-3-small").input("hi"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Auth(_)));
}
