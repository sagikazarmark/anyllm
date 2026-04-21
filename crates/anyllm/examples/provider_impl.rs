//! # Implementing a provider crate for `anyllm`
//!
//! This example is the minimal skeleton a new provider crate builds
//! around. It covers the four things every provider has to get right:
//!
//! 1. **Trait contract.** Implement [`ProviderIdentity`] and
//!    [`ChatProvider`], plus [`EmbeddingProvider`] if the upstream
//!    exposes embeddings. The `extract` feature's `ExtractExt` is
//!    automatic once `ChatProvider` is in place; no separate impl
//!    block is needed unless the provider wants to customize it.
//! 2. **Capability reporting.** Answer [`CapabilitySupport`] per
//!    [`ChatCapability`] per model. Classify by model-family prefix
//!    when the provider has uniform behavior across a family, and
//!    return `Unknown` where the answer depends on a variant the
//!    crate does not enumerate. Callers that know their model can
//!    override via `with_chat_capabilities`.
//! 3. **Error mapping.** Translate upstream failure modes into the
//!    right [`anyllm::Error`] variant so [`RetryingChatProvider`],
//!    [`FallbackChatProvider`], and the tracing wrapper react
//!    correctly.
//! 4. **Conformance testing.** Hook the provider into
//!    `anyllm-conformance` for fixture-based request, response, and
//!    error coverage.
//!
//! Run with `cargo run --example provider_impl`.
//!
//! ## Error mapping cheat sheet
//!
//! | Upstream signal | `anyllm::Error` variant |
//! |-----------------|-------------------------|
//! | 401 / 403 / bad API key | `Auth` |
//! | 429 + retry-after | `RateLimited { retry_after, request_id }` |
//! | 503 or provider-native overloaded | `Overloaded { retry_after, request_id }` |
//! | 5xx or no HTTP status | `Provider { status, message, body, request_id }` |
//! | 4xx (non-auth) | `InvalidRequest` |
//! | Transport timeout | `Timeout` |
//! | Deserialization failure | `Serialization` |
//! | Unknown model name | `ModelNotFound` |
//! | Prompt exceeded context window | `ContextLengthExceeded` |
//! | Safety-filtered output | `ContentFiltered` |
//!
//! Populate `request_id` on the retryable variants (`RateLimited`,
//! `Overloaded`, `Provider`) from the upstream response headers so
//! production logs stay greppable. The HTTP-native providers share
//! `anyllm_openai_compat::map_http_error` as a reusable mapper; real
//! implementations in this repo (`anyllm-openai`, `anyllm-anthropic`,
//! `anyllm-gemini`) call it or a sibling function with the same shape.
//!
//! ## Conformance harness
//!
//! `anyllm-conformance` ships fixture helpers for provider authors:
//!
//! ```toml
//! [dev-dependencies]
//! anyllm-conformance = { path = "../anyllm-conformance" }
//! ```
//!
//! A `conformance_tests.rs` module calls helpers like
//! `assert_json_fixture_eq`, `assert_response_fixture_eq`, and
//! `assert_error_fixture_eq` against JSON fixtures in a `fixtures/`
//! directory. The provider crates in this repo follow this layout
//! uniformly; `anyllm-openai/src/conformance_tests.rs` is a good
//! starting point.

use anyllm::ProviderIdentity;
use anyllm::prelude::*;

/// A trivial in-memory provider used to demonstrate the trait contract.
///
/// Real providers hold an HTTP client, credentials, and a capability
/// resolver hook here. See `anyllm-openai` for a reference implementation.
struct StaticProvider;

impl ProviderIdentity for StaticProvider {
    fn provider_name(&self) -> &'static str {
        "static-demo"
    }
}

impl ChatProvider for StaticProvider {
    type Stream = SingleResponseStream;

    async fn chat(&self, request: &ChatRequest) -> anyllm::Result<ChatResponse> {
        // Demonstrate error mapping: unknown models surface as
        // `ModelNotFound` (non-retryable) rather than a generic
        // `Provider` error. Application code can pattern-match the
        // variant and skip retry/fallback on it.
        if !request.model.starts_with("demo-") {
            return Err(anyllm::Error::ModelNotFound(format!(
                "static-demo does not know model '{}'",
                request.model
            )));
        }

        Ok(ChatResponse::new(vec![ContentBlock::Text {
            text: format!("hello from {}", self.provider_name()),
        }])
        .finish_reason(FinishReason::Stop)
        .model(request.model.clone()))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> anyllm::Result<Self::Stream> {
        // Non-native streaming: build the full response, then wrap it in
        // a `SingleResponseStream` that replays it as normalized events.
        // A provider with true server-side streaming would report
        // `NativeStreaming: Supported` in `chat_capability`.
        Ok(self.chat(request).await?.into_stream())
    }

    fn chat_capability(&self, _model: &str, capability: ChatCapability) -> CapabilitySupport {
        // Return `Unknown` for capabilities we do not explicitly
        // classify. Callers with a concrete answer can install a
        // resolver via `with_chat_capabilities` to override.
        match capability {
            ChatCapability::Streaming => CapabilitySupport::Supported,
            ChatCapability::NativeStreaming => CapabilitySupport::Unsupported,
            _ => CapabilitySupport::Unknown,
        }
    }
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let provider = StaticProvider;

    // Happy path.
    let ok = ChatRequest::new("demo-model")
        .system("You are concise.")
        .user("Say hello");
    let response = provider.chat(&ok).await?;
    println!("chat text: {}", response.text_or_empty());
    println!(
        "streaming support: {:?}",
        provider.chat_capability(&ok.model, ChatCapability::Streaming)
    );
    println!(
        "native streaming: {:?}",
        provider.chat_capability(&ok.model, ChatCapability::NativeStreaming)
    );

    // Stream collection works even though streaming is non-native:
    // `SingleResponseStream` replays the full response as normalized
    // events so callers get the same shape from either entry point.
    let streamed = provider.chat_stream(&ok).await?.collect_response().await?;
    println!("stream text: {}", streamed.text_or_empty());

    // Error mapping: unknown models surface as `ModelNotFound`.
    let bad = ChatRequest::new("unknown-model").user("Say hello");
    match provider.chat(&bad).await {
        Ok(_) => unreachable!("unknown-model should error"),
        Err(anyllm::Error::ModelNotFound(message)) => {
            println!("error mapping: ModelNotFound: {message}");
        }
        Err(other) => println!("unexpected error variant: {other:?}"),
    }

    Ok(())
}
