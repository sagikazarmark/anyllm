//! Anthropic prompt caching with a cached preamble + volatile tail.
//!
//! Demonstrates the motivating use case for carrying multiple
//! [`SystemPrompt`](anyllm::SystemPrompt) entries with distinct typed
//! options on a [`ChatRequest`](anyllm::ChatRequest): a large, stable
//! preamble that should hit the Anthropic prompt cache, followed by a
//! small per-request instruction that should not be cached.
//!
//! Anthropic lets callers mark individual system-prompt segments with a
//! `cache_control` hint so the server can memoize the tokens up to that
//! point and reuse them across requests. In `anyllm` this hint is
//! expressed as the typed option [`anyllm_anthropic::CacheControl`]
//! attached via [`SystemPrompt::with_option`](anyllm::SystemPrompt::with_option).
//! Prompts without the option are sent as ordinary, non-cached segments.
//!
//! Putting the stable, long preamble first (cached) and the volatile
//! per-request tail afterwards (not cached) lets Anthropic reuse the
//! cached prefix while still honoring fresh instructions on every call.
//!
//! Run:
//!     ANTHROPIC_API_KEY=... cargo run -p anyllm-examples --example anthropic_prompt_caching
//!
//! Without `ANTHROPIC_API_KEY` set, the example exits with a friendly
//! message rather than panicking.

use anyllm::prelude::*;
use anyllm_anthropic::{CacheControl, Provider as Anthropic};

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .is_none()
    {
        eprintln!("Set ANTHROPIC_API_KEY to run this example.");
        return Ok(());
    }

    // Construct the Anthropic provider from the standard env vars
    // (`ANTHROPIC_API_KEY`, optional `ANTHROPIC_BASE_URL`).
    let provider = Anthropic::from_env()?;

    let model = std::env::var("ANTHROPIC_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());

    // A cached preamble — typically long and stable across many
    // requests. In real usage this would be substantially larger
    // (Anthropic only caches segments that exceed a minimum token
    // threshold), but the shape of the request is identical.
    let cached_preamble = "You are a senior Rust reviewer. Follow these style rules: \
        prefer explicit error handling, avoid unwrap() outside tests, \
        keep public functions documented, and reason about ownership \
        before suggesting borrow changes. \
        Always cite the exact line a comment refers to and keep feedback \
        grounded in the diff rather than speculation. \
        [In real code this preamble would be much longer so it exceeds \
        the Anthropic cache threshold.]";

    // A small, per-request volatile tail — only meaningful for this one
    // turn, so it intentionally does NOT carry a CacheControl option.
    let volatile_tail = format!(
        "Context for this review: PR #{pr} touches concurrency primitives.",
        pr = 42,
    );

    // Two system prompts: the first cached, the second volatile. The
    // Anthropic provider preserves this ordering end-to-end so the wire
    // payload marks only the preamble with `cache_control`.
    let request = ChatRequest::new(model)
        .system(SystemPrompt::new(cached_preamble).with_option(CacheControl::ephemeral()))
        .system(SystemPrompt::new(volatile_tail))
        .user("Please review the attached diff and list the top three concerns.");

    let response = provider.chat(&request).await?;

    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }

    println!("{}", response.text_or_empty());
    Ok(())
}
