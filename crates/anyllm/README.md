# anyllm

[![crates.io](https://img.shields.io/crates/v/anyllm?style=flat-square)](https://crates.io/crates/anyllm)
[![docs.rs](https://img.shields.io/docsrs/anyllm?style=flat-square)](https://docs.rs/anyllm)

**Provider-agnostic LLM abstractions for Rust.**

`anyllm` is a low-level crate for code that wants one interface for LLM APIs
(both chat and embeddings) while leaving HTTP transport, request translation,
and response parsing to provider implementations.

Use it when you want to:

- write application code against one provider-neutral contract
- implement a provider adapter on top of shared request and response types
- normalize streaming, tool calls, and metadata across providers
- produce embedding vectors from multiple providers with a single trait

`anyllm` is intentionally not a provider client and not an agent framework.

## You May Be Looking For

- API docs: <https://docs.rs/anyllm>
- `ChatProvider`: <https://docs.rs/anyllm/latest/anyllm/trait.ChatProvider.html>
- `EmbeddingProvider`: <https://docs.rs/anyllm/latest/anyllm/trait.EmbeddingProvider.html>
- chat example: <https://github.com/sagikazarmark/anyllm/blob/main/crates/anyllm/examples/chat.rs>
- embedding example: <https://github.com/sagikazarmark/anyllm/blob/main/crates/anyllm/examples/embedding.rs>
- streaming example: <https://github.com/sagikazarmark/anyllm/blob/main/crates/anyllm/examples/stream.rs>
- provider implementation example: <https://github.com/sagikazarmark/anyllm/blob/main/crates/anyllm/examples/provider_impl.rs>

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

From the workspace root:

```bash
cargo run -p anyllm --example chat --features mock
```

## Features

Default features are empty.

| Feature | What it enables |
| --- | --- |
| `extract` | structured extraction helpers such as `ExtractExt`, `Extractor`, and `ExtractingProvider` |
| `mock` | deterministic mock providers (`MockProvider`, `MockStreamingProvider`, `MockEmbeddingProvider`) and response builders for tests and examples |
| `tracing` | tracing-based instrumentation via `TracingChatProvider` |

## Implementing A Provider

If you are writing a provider, implement `ProviderIdentity` first, then whichever
capability traits apply: `ChatProvider`, `EmbeddingProvider`, or both.

Chat providers with native streaming return their own concrete stream type from
`chat_stream()`. Providers without native streaming can return
`SingleResponseStream` and still participate in the same streaming contract.

Embedding providers implement a single batch-oriented `embed()` method.

If you need a working skeleton, start here:

- <https://github.com/sagikazarmark/anyllm/blob/main/crates/anyllm/examples/provider_impl.rs>
