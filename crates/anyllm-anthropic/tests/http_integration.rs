use anyllm::{ChatProvider, ChatRequest};
use anyllm_anthropic::{ChatRequestOptions, Provider};
use anyllm_conformance::{
    FixtureDir, MockHttpResponse, TestHttpServer, assert_partial_stream_fixture_eq,
    assert_response_fixture_eq, assert_stream_finish_error_fixture_eq, assert_stream_fixture_eq,
    load_json_fixture, load_text_fixture,
};
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
        .api_key("anthropic-test-key")
        .base_url(format!("{}/", server.url()))
        .build()
        .unwrap();

    let response = provider
        .chat(
            &ChatRequest::new("claude-sonnet-4-20250514")
                .user("Hello from the transport integration test.")
                .with_option(ChatRequestOptions {
                    anthropic_beta: vec!["integration-beta".into()],
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
    assert_eq!(request.path, "/v1/messages");
    assert_eq!(request.header("x-api-key"), Some("anthropic-test-key"));
    assert_eq!(request.header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(request.header("anthropic-beta"), Some("integration-beta"));
    assert_eq!(request.header("content-type"), Some("application/json"));
    assert_eq!(body["model"], "claude-sonnet-4-20250514");
    assert_eq!(body["messages"][0]["role"], "user");
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
        .api_key("anthropic-test-key")
        .base_url(format!("{}/", server.url()))
        .build()
        .unwrap();

    let stream = provider
        .chat_stream(&ChatRequest::new("claude-sonnet-4-20250514").user("Stream this."))
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

    assert_eq!(request.path, "/v1/messages");
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
        .api_key("anthropic-test-key")
        .base_url(format!("{}/", server.url()))
        .build()
        .unwrap();

    let stream = provider
        .chat_stream(&ChatRequest::new("claude-sonnet-4-20250514").user("Stream this."))
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
        .chat_stream(&ChatRequest::new("claude-sonnet-4-20250514").user("Stream this."))
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
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "too many requests"
            }
        }),
    )
    .with_header("x-request-id", "req-anthropic-429")
    .with_header("retry-after", "3")])
    .await;

    let provider = Provider::builder()
        .api_key("anthropic-test-key")
        .base_url(format!("{}/", server.url()))
        .build()
        .unwrap();

    let err = provider
        .chat(&ChatRequest::new("claude-sonnet-4-20250514").user("Trigger an error."))
        .await
        .unwrap_err();

    match err {
        anyllm::Error::RateLimited {
            message,
            retry_after,
            request_id,
        } => {
            assert_eq!(message, "too many requests");
            assert_eq!(retry_after, Some(std::time::Duration::from_secs(3)));
            assert_eq!(request_id.as_deref(), Some("req-anthropic-429"));
        }
        other => panic!("expected RateLimited, got {other:?}"),
    }
}

#[tokio::test]
async fn chat_ignores_invalid_retry_after_header() {
    let server = TestHttpServer::spawn([MockHttpResponse::json(
        429,
        &json!({
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "too many requests"
            }
        }),
    )
    .with_header("retry-after", "-1")])
    .await;

    let provider = Provider::builder()
        .api_key("anthropic-test-key")
        .base_url(format!("{}/", server.url()))
        .build()
        .unwrap();

    let err = provider
        .chat(&ChatRequest::new("claude-sonnet-4-20250514").user("Trigger an error."))
        .await
        .unwrap_err();

    match err {
        anyllm::Error::RateLimited { retry_after, .. } => {
            assert_eq!(retry_after, None);
        }
        other => panic!("expected RateLimited, got {other:?}"),
    }
}
