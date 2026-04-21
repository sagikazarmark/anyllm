# anyllm

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/sagikazarmark/anyllm/ci.yaml?style=flat-square)](https://github.com/sagikazarmark/anyllm/actions/workflows/ci.yaml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/sagikazarmark/anyllm/badge?style=flat-square)](https://securityscorecards.dev/viewer/?uri=github.com/sagikazarmark/anyllm)
[![crates.io](https://img.shields.io/crates/v/anyllm?style=flat-square)](https://crates.io/crates/anyllm)
[![docs.rs](https://img.shields.io/docsrs/anyllm?style=flat-square)](https://docs.rs/anyllm)

**Provider-agnostic LLM abstractions and adapters for Rust.**

`anyllm` lets you build against LLM APIs (chat and embeddings) without
hard-coding the rest of your application to one provider SDK.

The core of the project is the [`anyllm`](crates/anyllm) crate: a small, low-level
abstraction layer with shared request and response types for both chat (streaming,
tool calls, structured output) and embeddings, plus a handful of wrappers.

The intent is to keep application-facing code portable and keep
provider-specific request translation, transport, and parsing in provider
crates.

That gives applications and libraries one interface for model interaction while
leaving provider-specific details at the edges.

`anyllm` is intentionally a building block, not an agent framework or
orchestration runtime.

## Where To Start

- use `anyllm` in application code: [`anyllm`](crates/anyllm)
- implement a provider: [`crates/anyllm/examples/provider_impl.rs`](crates/anyllm/examples/provider_impl.rs)
- explore the API: [`docs.rs/anyllm`](https://docs.rs/anyllm)

## Workspace Crates

| Crate | Role | Notes |
| --- | --- | --- |
| [`anyllm`](crates/anyllm) | Core abstraction | Shared chat + embedding request/response types, streaming, tools, and wrappers |
| [`anyllm-conformance`](crates/anyllm-conformance) | Test support | Fixture-based conformance helpers, shared behavioral contract assertions, and a local mock HTTP server for provider crates |
| [`anyllm-openai`](crates/anyllm-openai) | Provider adapter | OpenAI chat + embedding provider built on the shared `anyllm` surface |
| [`anyllm-anthropic`](crates/anyllm-anthropic) | Provider adapter | Anthropic Messages API chat provider |
| [`anyllm-gemini`](crates/anyllm-gemini) | Provider adapter | Google Gemini chat + embedding provider |
| [`anyllm-openai-compat`](crates/anyllm-openai-compat) | Provider toolkit | Reusable transport and normalization helpers for OpenAI-compatible providers (Cloudflare, etc.), with chat + embedding |
| [`anyllm-cloudflare-worker`](crates/anyllm-cloudflare-worker) | Provider adapter | Cloudflare Workers AI via the native `worker::Ai` binding (use from inside a Worker; no outbound HTTP) |

## Example

This example uses the built-in mock provider, so it runs without credentials.

```toml
[dependencies]
anyllm = { version = "0.1", features = ["mock"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use anyllm::prelude::*;

fn build_provider() -> MockProvider {
    MockProvider::build(|builder| builder.text("Deterministic hello from anyllm."))
}

fn build_request() -> ChatRequest {
    ChatRequest::new("demo-model").user("Say hello")
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let provider = build_provider();
    let request = build_request();
    let response = provider.chat(&request).await?;

    println!("chat text: {}", response.text_or_empty());
    Ok(())
}
```

Run it with:

```bash
cargo run -p anyllm --example chat --features mock
```

## Providers

| Provider | Crate | Chat | Embeddings | Notes |
| --- | --- | --- | --- | --- |
| OpenAI | [`anyllm-openai`](crates/anyllm-openai) | ✓ | ✓ | Streaming, tools, structured output; `/v1/embeddings` with optional dimensions |
| Anthropic | [`anyllm-anthropic`](crates/anyllm-anthropic) | ✓ | n/a | Messages API with streaming, tools, and reasoning; embeddings are out of scope (Voyage ships separately) |
| Gemini | [`anyllm-gemini`](crates/anyllm-gemini) | ✓ | ✓ | `generateContent`/`streamGenerateContent` for chat; `batchEmbedContents` for embeddings |
| OpenAI-compatible (Groq, Cloudflare, etc.) | [`anyllm-openai-compat`](crates/anyllm-openai-compat) | ✓ | ✓ | Toolkit plus presets for any OpenAI-compatible endpoint; HTTP-based |
| Cloudflare Workers AI | [`anyllm-cloudflare-worker`](crates/anyllm-cloudflare-worker) | ✓ | ✓ | Native `worker::Ai` binding for code already running inside a Cloudflare Worker |

## License

The project is licensed under the [MIT License](LICENSE).
