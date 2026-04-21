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

Embeddings are not in scope for this adapter: Anthropic does not expose an
embeddings API directly, and Voyage AI (the partner model provider) ships
through its own SDK. If you need embeddings alongside Claude chat, pair this
crate with an embedding-capable adapter such as `anyllm-openai`,
`anyllm-gemini`, or `anyllm-openai-compat`.

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

- Anthropic-specific request knobs live in typed `ChatRequestOptions` rather
  than top-level untyped JSON.
- Anthropic does not support `ChatRequest.response_format`; use plain text or
  the shared extraction fallback path instead.
- Reasoning signatures and reasoning text are preserved in streamed and
  collected responses.
