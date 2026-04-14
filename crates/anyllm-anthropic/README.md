# anyllm-anthropic

Anthropic provider adapter for `anyllm`.

## Use it when

- you want Claude models behind the shared `anyllm::ChatProvider` trait
- you want streamed reasoning/thinking blocks surfaced in normalized response content
- you want Anthropic-specific API details hidden behind the common request/response model

## Capabilities

`anyllm-anthropic` currently reports:

- tools
- streaming
- vision
- reasoning blocks

It does not currently report native structured-output support, so cross-provider
typed extraction uses the non-native fallback path where appropriate.

## Quick start

```toml
[dependencies]
anyllm = "0.1.0"
anyllm-anthropic = "0.1.0"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use anyllm::prelude::*;
use anyllm_anthropic::Provider;

#[tokio::main]
async fn main() -> Result<()> {
    let provider = Provider::from_env()?;

    let response = provider
        .chat(&ChatRequest::new("claude-sonnet-4-20250514")
            .user("Summarize why Rust is useful for agents."))
        .await?;

    println!("{}", response.text_or_empty());
    Ok(())
}
```

Environment variables:

- required: `ANTHROPIC_API_KEY`
- optional: `ANTHROPIC_BASE_URL`

## Notes

- Use `Provider::new_with_base_url(...)` or `Provider::builder()` when you need
  a custom base URL or HTTP client.
- The default transport applies a 120 second timeout to non-streaming requests
  only. Streaming requests intentionally avoid a whole-request wall clock
  timeout so long-lived SSE responses are not cut off mid-stream.
- Inject your own `reqwest::Client` with `Provider::builder().client(...)` when
  you need a different transport policy.
- Anthropic reasoning signatures and reasoning text are preserved in streamed and
  collected responses.
- Anthropic-specific request knobs live in typed `ChatRequestOptions` rather
  than top-level untyped JSON.
- Anthropic does not support `ChatRequest.response_format`; use plain text or the
  shared extraction fallback path instead.
- Tool-result replay is text-only on this adapter; multimodal tool results are
  rejected instead of being silently degraded.
- The adapter currently targets the Anthropic content block types exercised by
  this repo's conformance and integration tests. Unknown newer block types are
  not normalized yet.
- If `max_tokens` is omitted, the adapter sends Anthropic's default output budget
  of `4096`, and reasoning budgets may bump that ceiling upward to satisfy the
  API's `budget_tokens < max_tokens` requirement.
