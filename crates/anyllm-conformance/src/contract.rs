//! Provider-agnostic behavioral contract helpers.
//!
//! Each helper takes any [`ChatProvider`] plus a pre-built [`ChatRequest`]
//! and verifies that the provider's response honors the portable contract:
//! the finish reason is the right variant, tool-call blocks carry valid
//! JSON arguments, streaming events reconstruct into a coherent response,
//! and so on.
//!
//! The helpers are provider-agnostic by construction. Provider crates wire
//! a [`TestHttpServer`](crate::TestHttpServer) with provider-specific
//! fixture responses, build their `Provider` against the server's URL,
//! and then call these helpers so the same behavioral contract is
//! enforced uniformly across the provider matrix.
//!
//! For live-API checks that build the request internally, see
//! [`crate::e2e`].
//!
//! # Example
//!
//! ```no_run
//! use anyllm::prelude::*;
//! use anyllm_conformance::{TestHttpServer, contract::assert_chat_success_contract};
//!
//! # async fn example<P: ChatProvider>(provider: &P, server: &TestHttpServer) {
//! let request = ChatRequest::new("my-model").user("say hi");
//! // `server` has been configured with the provider's success fixture
//! // and `provider` is wired to `server.base_url()`.
//! assert_chat_success_contract(provider, &request).await;
//! # let _ = server;
//! # }
//! ```

use anyllm::{ChatProvider, ChatRequest, ChatStreamExt, FinishReason};

/// Assert a successful non-streaming chat: response has text, Stop finish
/// reason, and non-empty content.
pub async fn assert_chat_success_contract(provider: &impl ChatProvider, request: &ChatRequest) {
    let response = provider
        .chat(request)
        .await
        .expect("chat_success: chat request failed");
    let text = response
        .text()
        .expect("chat_success: response has no text block");
    assert!(!text.is_empty(), "chat_success: response text is empty");
    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "chat_success: expected Stop finish reason, got {:?}",
        response.finish_reason
    );
}

/// Assert a successful streaming chat: events arrive, the collector
/// reconstructs a response with text and a Stop finish reason.
pub async fn assert_streaming_chat_contract(provider: &impl ChatProvider, request: &ChatRequest) {
    let stream = provider
        .chat_stream(request)
        .await
        .expect("streaming_chat: chat_stream request failed");
    let response = stream
        .collect_response()
        .await
        .expect("streaming_chat: stream collection failed");
    let text = response
        .text()
        .expect("streaming_chat: response has no text block");
    assert!(!text.is_empty(), "streaming_chat: response text is empty");
    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "streaming_chat: expected Stop finish reason, got {:?}",
        response.finish_reason
    );
}

/// Assert a non-streaming tool-calling round trip: the response has tool
/// calls, each call's arguments parse as JSON, and the finish reason is
/// [`FinishReason::ToolCalls`].
pub async fn assert_tool_call_contract(provider: &impl ChatProvider, request: &ChatRequest) {
    let response = provider
        .chat(request)
        .await
        .expect("tool_call: chat request failed");
    assert!(
        response.has_tool_calls(),
        "tool_call: response has no tool calls"
    );
    assert_eq!(
        response.finish_reason,
        Some(FinishReason::ToolCalls),
        "tool_call: expected ToolCalls finish reason, got {:?}",
        response.finish_reason
    );
    for call in response.tool_calls() {
        assert!(!call.name.is_empty(), "tool_call: empty tool name");
        assert!(
            serde_json::from_str::<serde_json::Value>(call.arguments).is_ok(),
            "tool_call: arguments not valid JSON for {}: {}",
            call.name,
            call.arguments
        );
    }
}

/// Assert a streaming tool-calling round trip: the stream reconstructs
/// into a response with tool calls, each call's arguments parse as JSON,
/// and the finish reason is [`FinishReason::ToolCalls`].
///
/// A reconstructed tool call is evidence that the stream emitted the
/// underlying `BlockStart(ToolCall)` / `ToolCallDelta` / `BlockStop`
/// sequence; the collector would otherwise reject the event shape.
pub async fn assert_streaming_tool_call_contract(
    provider: &impl ChatProvider,
    request: &ChatRequest,
) {
    let stream = provider
        .chat_stream(request)
        .await
        .expect("streaming_tool_call: chat_stream request failed");
    let response = stream
        .collect_response()
        .await
        .expect("streaming_tool_call: stream collection failed");
    assert!(
        response.has_tool_calls(),
        "streaming_tool_call: collected response has no tool calls"
    );
    assert_eq!(
        response.finish_reason,
        Some(FinishReason::ToolCalls),
        "streaming_tool_call: expected ToolCalls finish reason, got {:?}",
        response.finish_reason
    );
    for call in response.tool_calls() {
        assert!(
            !call.name.is_empty(),
            "streaming_tool_call: empty tool name"
        );
        assert!(
            serde_json::from_str::<serde_json::Value>(call.arguments).is_ok(),
            "streaming_tool_call: arguments not valid JSON for {}: {}",
            call.name,
            call.arguments
        );
    }
}

/// Assert a structured-output response: the text is valid JSON.
pub async fn assert_structured_output_contract(
    provider: &impl ChatProvider,
    request: &ChatRequest,
) {
    let response = provider
        .chat(request)
        .await
        .expect("structured_output: chat request failed");
    let text = response
        .text()
        .expect("structured_output: response has no text block");
    serde_json::from_str::<serde_json::Value>(&text).unwrap_or_else(|e| {
        panic!("structured_output: response text is not valid JSON: {e}\nraw: {text}")
    });
}

/// Assert that a request against an authentication-failure fixture
/// surfaces as [`anyllm::Error::Auth`].
pub async fn assert_auth_error_contract(provider: &impl ChatProvider, request: &ChatRequest) {
    let err = provider
        .chat(request)
        .await
        .expect_err("auth_error: expected chat to fail with Auth error");
    assert!(
        matches!(err, anyllm::Error::Auth(_)),
        "auth_error: expected Error::Auth, got {err:?}"
    );
    assert!(
        !err.is_retryable(),
        "auth_error: Auth should not be is_retryable()"
    );
}

/// Assert that a request against a rate-limited-failure fixture surfaces
/// as [`anyllm::Error::RateLimited`] and classifies as retryable and
/// transient.
pub async fn assert_rate_limited_error_contract(
    provider: &impl ChatProvider,
    request: &ChatRequest,
) {
    let err = provider
        .chat(request)
        .await
        .expect_err("rate_limited: expected chat to fail with RateLimited error");
    assert!(
        matches!(err, anyllm::Error::RateLimited { .. }),
        "rate_limited: expected Error::RateLimited, got {err:?}"
    );
    assert!(
        err.is_retryable(),
        "rate_limited: RateLimited should be is_retryable()"
    );
    assert!(
        err.is_transient(),
        "rate_limited: RateLimited should be is_transient()"
    );
}

/// Assert that a request against an overloaded-failure fixture surfaces
/// as [`anyllm::Error::Overloaded`] and classifies as retryable and
/// transient.
pub async fn assert_overloaded_error_contract(provider: &impl ChatProvider, request: &ChatRequest) {
    let err = provider
        .chat(request)
        .await
        .expect_err("overloaded: expected chat to fail with Overloaded error");
    assert!(
        matches!(err, anyllm::Error::Overloaded { .. }),
        "overloaded: expected Error::Overloaded, got {err:?}"
    );
    assert!(
        err.is_retryable(),
        "overloaded: Overloaded should be is_retryable()"
    );
    assert!(
        err.is_transient(),
        "overloaded: Overloaded should be is_transient()"
    );
}

/// Assert that a request against a context-length-exceeded fixture
/// surfaces as [`anyllm::Error::ContextLengthExceeded`].
pub async fn assert_context_length_exceeded_contract(
    provider: &impl ChatProvider,
    request: &ChatRequest,
) {
    let err = provider
        .chat(request)
        .await
        .expect_err("context_length: expected chat to fail with ContextLengthExceeded");
    assert!(
        matches!(err, anyllm::Error::ContextLengthExceeded { .. }),
        "context_length: expected Error::ContextLengthExceeded, got {err:?}"
    );
    assert!(
        !err.is_retryable(),
        "context_length: ContextLengthExceeded should not be is_retryable()"
    );
}
