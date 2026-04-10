# anyllm

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/sagikazarmark/anyllm/ci.yaml?style=flat-square)](https://github.com/sagikazarmark/anyllm/actions/workflows/ci.yaml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/sagikazarmark/anyllm/badge?style=flat-square)](https://securityscorecards.dev/viewer/?uri=github.com/sagikazarmark/anyllm)
[![crates.io](https://img.shields.io/crates/v/anyllm?style=flat-square)](https://crates.io/crates/anyllm)
[![docs.rs](https://img.shields.io/docsrs/anyllm?style=flat-square)](https://docs.rs/anyllm)

**Provider-agnostic LLM abstractions and adapters for Rust.**

`anyllm` allows building against chat-style LLM APIs without
hard-coding the rest of your application to one provider SDK.

The core of the project is the [`anyllm`](anyllm) crate: a small, low-level
abstraction layer for shared request and response types, streaming, tool calls,
and a few wrappers.

The intent is to keep application-facing code portable and keep
provider-specific request translation, transport, and parsing in provider
crates.

That gives applications and libraries one interface for model interaction while
leaving provider-specific details at the edges.

`anyllm` is intentionally a building block, not an agent framework or
orchestration runtime.

## Where To Start

- use `anyllm` in application code: [`anyllm`](anyllm)
- implement a provider: [`anyllm/examples/provider_impl.rs`](anyllm/examples/provider_impl.rs)
- explore the API: [`docs.rs/anyllm`](https://docs.rs/anyllm)

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

| Provider | Crate | Status | Notes |
| --- | --- | --- | --- |
| TBD | `-` | `-` | Provider list will be filled in later |

## License

The project is licensed under the [MIT License](LICENSE).
