//! Cross-provider fallback example: [`FallbackChatProvider`] tries the
//! primary provider first and falls back to a secondary on retryable
//! errors. Non-retryable errors (Auth, InvalidRequest, ModelNotFound,
//! Serialization, UnexpectedResponse) surface directly from the primary
//! without the fallback being called; there is no point retrying the
//! same bad input against a different upstream.
//!
//! Requires credentials for both providers:
//!
//! - `OPENAI_API_KEY` (primary)
//! - `ANTHROPIC_API_KEY` (fallback)
//!
//! Note: the request's model name is passed to whichever provider
//! actually handles it. In production you would normalize model names
//! across providers at the application layer.

use anyllm::prelude::*;
use anyllm_anthropic::Provider as Anthropic;
use anyllm_openai::Provider as OpenAI;

fn error_shape(err: &Error) -> &'static str {
    match err {
        Error::Auth(_) => "Auth (non-retryable; fallback skipped)",
        Error::InvalidRequest(_) => "InvalidRequest (non-retryable; fallback skipped)",
        Error::ModelNotFound(_) => "ModelNotFound (non-retryable; fallback skipped)",
        Error::Serialization(_) => "Serialization (non-retryable; fallback skipped)",
        Error::UnexpectedResponse(_) => "UnexpectedResponse (non-retryable; fallback skipped)",
        Error::Fallback { .. } => "Fallback (primary retryable, fallback also failed)",
        Error::Timeout(_) => "Timeout (retryable)",
        Error::RateLimited { .. } => "RateLimited (retryable)",
        Error::Overloaded { .. } => "Overloaded (retryable)",
        Error::Provider { .. } => "Provider (retryable)",
        _ => "other",
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "In one sentence: why is Rust good for servers?".into());

    let primary = OpenAI::from_env()?;
    let fallback = Anthropic::from_env()?;
    let provider = FallbackChatProvider::new(primary, fallback);

    eprintln!("primary=openai (gpt-4o), fallback=anthropic (claude-sonnet-4-20250514)");

    // Happy path: the primary responds and the fallback is never called.
    let ok_request = ChatRequest::new("gpt-4o").user(prompt);
    let response = provider.chat(&ok_request).await?;
    eprintln!("happy path finish_reason={:?}", response.finish_reason);
    println!("happy path: {}", response.text_or_empty());

    // Auth-filter demo: a model the primary does not recognize surfaces
    // as Error::ModelNotFound from the primary. The fallback is NOT
    // invoked because ModelNotFound is non-retryable; there is no reason
    // to expect a different provider to recognize the same bad name.
    let bad_request = ChatRequest::new("this-model-does-not-exist").user("Hello");
    match provider.chat(&bad_request).await {
        Ok(_) => eprintln!("unexpected: fake model did not error"),
        Err(err) => {
            eprintln!("auth-filter demo: {}", error_shape(&err));
            eprintln!("  is_retryable={}", err.is_retryable());
        }
    }

    Ok(())
}
