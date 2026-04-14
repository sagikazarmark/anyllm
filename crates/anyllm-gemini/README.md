# anyllm-gemini

Google Gemini provider adapter for `anyllm`.

## Use it when

- you want Gemini models behind the shared `anyllm::ChatProvider` trait
- you want reasoning blocks and Gemini-native parallel tool calls normalized into the common contract
- you want structured outputs plus cross-provider request/response types

## Capabilities

`anyllm-gemini` currently reports:

- tools
- streaming
- vision
- structured output
- reasoning blocks
- parallel tool calls

## Quick start

```toml
[dependencies]
anyllm = "0.1.0"
anyllm-gemini = "0.1.0"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use anyllm::prelude::*;
use anyllm_gemini::Provider;

#[tokio::main]
async fn main() -> Result<()> {
    let provider = Provider::from_env()?;

    let response = provider
        .chat(&ChatRequest::new("gemini-2.5-pro")
            .user("Explain tool calling in two sentences."))
        .await?;

    println!("{}", response.text_or_empty());
    Ok(())
}
```

Environment variables:

- required: `GEMINI_API_KEY`
- optional: `GEMINI_BASE_URL`

## Notes

- Use `Provider::new_with_base_url(...)` or `Provider::builder()` when you need
  a custom base URL or HTTP client.
- The default client applies a 120 second timeout to non-streaming requests only.
  Streaming requests intentionally avoid a whole-request wall clock timeout so
  long-lived SSE responses are not cut off mid-stream.
- Gemini finish semantics are normalized so tool-call responses produce the
  common `FinishReason::ToolCalls` contract.
- Reasoning/thinking content is surfaced as `ContentBlock::Reasoning` when the
  model returns it.
- `ChatRequestOptions.candidate_count` currently supports only `1`; larger
  values are rejected until `anyllm` grows a multi-candidate response shape.
- Prefer portable `ChatRequest.response_format(...)` over
  `ChatRequestOptions.response_mime_type`. The provider-specific MIME override is
  an escape hatch for advanced Gemini-only cases.
- Image URLs are converted through Gemini's file-data path using best-effort
  MIME inference from the URL extension, so extensionless signed URLs are not
  accepted yet.
