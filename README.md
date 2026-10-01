# allama

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/sagikazarmark/allama/ci.yaml?style=flat-square)](https://github.com/sagikazarmark/allama/actions/workflows/ci.yaml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/sagikazarmark/allama/badge?style=flat-square)](https://securityscorecards.dev/viewer/?uri=github.com/sagikazarmark/allama)
[![crates.io](https://img.shields.io/crates/v/allama?style=flat-square)](https://crates.io/crates/allama)
[![docs.rs](https://img.shields.io/docsrs/allama?style=flat-square)](https://docs.rs/allama)

**Provider-agnostic LLM abstractions and adapters for Rust.**

`allama` lets you build against LLM APIs (chat and embeddings) with one
portable contract, and pair it with a provider crate for request
translation and transport. It is a building block, not an agent framework.

## Workspace Crates

| Crate | Role | Notes |
| --- | --- | --- |
| [`allama`](crates/allama) | Core abstraction | Shared chat + embedding request/response types, streaming, tools, and wrappers |
| [`allama-conformance`](crates/allama-conformance) | Test support | Fixture-based conformance helpers, shared behavioral contract assertions, and a local mock HTTP server for provider crates |
| [`allama-openai`](crates/allama-openai) | Provider adapter | OpenAI chat + embedding provider built on the shared `allama` surface |
| [`allama-anthropic`](crates/allama-anthropic) | Provider adapter | Anthropic Messages API chat provider |
| [`allama-gemini`](crates/allama-gemini) | Provider adapter | Google Gemini chat + embedding provider |
| [`allama-openai-compat`](crates/allama-openai-compat) | Provider toolkit | Reusable transport and normalization helpers for OpenAI-compatible providers (Cloudflare, etc.), with chat + embedding |
| [`allama-cloudflare-worker`](crates/allama-cloudflare-worker) | Provider adapter | Cloudflare Workers AI via the native `worker::Ai` binding (use from inside a Worker; no outbound HTTP) |

## Example

This example uses the built-in mock provider, so it runs without credentials.

```toml
[dependencies]
allama = { version = "0.1", features = ["mock"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use allama::prelude::*;

fn build_provider() -> MockProvider {
    MockProvider::build(|builder| builder.text("Deterministic hello from allama."))
}

fn build_request() -> ChatRequest {
    ChatRequest::new("demo-model").user("Say hello")
}

#[tokio::main]
async fn main() -> allama::Result<()> {
    let provider = build_provider();
    let request = build_request();
    let response = provider.chat(&request).await?;

    println!("chat text: {}", response.text_or_empty());
    Ok(())
}
```

Run it with:

```bash
cargo run -p allama --example chat --features mock
```

## Providers

| Provider | Crate | Chat | Embeddings | Notes |
| --- | --- | --- | --- | --- |
| OpenAI | [`allama-openai`](crates/allama-openai) | ✓ | ✓ | Streaming, tools, structured output; `/v1/embeddings` with optional dimensions |
| Anthropic | [`allama-anthropic`](crates/allama-anthropic) | ✓ | n/a | Messages API with streaming, tools, and reasoning; embeddings are out of scope (Voyage ships separately) |
| Gemini | [`allama-gemini`](crates/allama-gemini) | ✓ | ✓ | `generateContent`/`streamGenerateContent` for chat; `batchEmbedContents` for embeddings |
| OpenAI-compatible (Groq, Cloudflare, etc.) | [`allama-openai-compat`](crates/allama-openai-compat) | ✓ | ✓ | Toolkit plus presets for any OpenAI-compatible endpoint; HTTP-based |
| Cloudflare Workers AI | [`allama-cloudflare-worker`](crates/allama-cloudflare-worker) | ✓ | ✓ | Native `worker::Ai` binding for code already running inside a Cloudflare Worker |

## License

The project is licensed under the [MIT License](LICENSE).
