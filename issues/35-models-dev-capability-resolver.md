# Add opt-in `models.dev` chat capability resolver for provider users

## Problem statement

Some providers currently return `CapabilitySupport::Unknown` for chat
capabilities that `anyllm-models` may have advisory per-model metadata
for, but the repository does not provide a reusable resolver that users
can plug into `with_chat_capabilities(...)`.

At the same time, capability overrides are currently single-resolver only.
If a user wants to combine multiple sources of capability answers
(`models.dev`, local overrides, application-specific policy), they must
manually write a composite resolver.

This issue adds an opt-in `models.dev`-backed chat resolver plus ordered
resolver composition support in `anyllm` core.

## Evidence

- `ChatCapabilityResolver` already exists and uses `Option<CapabilitySupport>`
  where `None` defers to provider builtin logic:
  `crates/anyllm/src/chat/provider.rs`
- first-party provider crates already expose `with_chat_capabilities(...)`
  and consult that resolver before builtin capability logic:
  - `crates/anyllm-openai/src/chat.rs`
  - `crates/anyllm-anthropic/src/chat.rs`
  - `crates/anyllm-gemini/src/chat.rs`
- current provider builtins still return `Unknown` for some capability
  queries:
  - OpenAI: `crates/anyllm-openai/src/lib.rs`
  - Gemini: `crates/anyllm-gemini/src/lib.rs`
- `anyllm-models` already exposes typed `models.dev` structs including
  `tool_call`, `structured_output`, `reasoning`, and `modalities`:
  - `crates/anyllm-models/src/model.rs`
  - `crates/anyllm-models/src/provider.rs`
  - `crates/anyllm-models/src/lib.rs`
- `anyllm` already has embedding capability concepts in core, so resolver
  composition should be designed consistently across chat and embeddings:
  `crates/anyllm/src/embedding/provider.rs`

## Proposed solution

### 1. Add an opt-in resolver feature to `anyllm-models`

Add a `resolver` feature to `anyllm-models` that enables a dependency on
`anyllm` and exposes a `models.dev`-backed chat capability resolver.

Example shape:

```toml
[features]
resolver = ["dep:anyllm"]

[dependencies]
anyllm = { workspace = true, optional = true }
```

The resolver lives in `anyllm-models`, not in `anyllm` core.

### 2. Add a chat resolver backed by provider snapshot data

`anyllm-models` exposes a resolver implementation that deserializes a
single provider entry from `models.dev` and answers chat capability
queries from that provider's model map.

Example shape:

```rust
use std::collections::HashMap;
use anyllm::{CapabilitySupport, ChatCapability, ChatCapabilityResolver};
use crate::Model;

pub struct ModelsDevResolver {
    models: HashMap<String, Model>,
}

impl ModelsDevResolver {
    pub fn new(models: HashMap<String, Model>) -> Self {
        Self { models }
    }

    pub fn from_provider_json(json: &str) -> serde_json::Result<Self> {
        let provider: crate::Provider = serde_json::from_str(json)?;
        Ok(Self::new(provider.models))
    }
}

impl ChatCapabilityResolver for ModelsDevResolver {
    fn chat_capability(
        &self,
        model: &str,
        capability: ChatCapability,
    ) -> Option<CapabilitySupport> {
        let model = self.models.get(model)?;

        match capability {
            ChatCapability::ToolCalls => Some(support_from_bool(model.tool_call)),
            ChatCapability::StructuredOutput => {
                model.structured_output.map(support_from_bool)
            }
            ChatCapability::ImageInput => {
                Some(support_from_bool(model.modalities.input.iter().any(|m| m == "image")))
            }
            ChatCapability::ImageOutput => {
                Some(support_from_bool(model.modalities.output.iter().any(|m| m == "image")))
            }
            _ => None,
        }
    }
}

fn support_from_bool(b: bool) -> CapabilitySupport {
    if b {
        CapabilitySupport::Supported
    } else {
        CapabilitySupport::Unsupported
    }
}
```

### 3. Resolver answers must stay conservative

The resolver must only return `Some(...)` when the mapping from
`models.dev` data to a portable `ChatCapability` is explicit and
semantically clear.

The resolver must return `None` when:

- the model is absent from the snapshot
- the relevant field is missing
- the mapping is semantically unclear
- there is no direct upstream field for the queried capability

`None` means "defer to the next resolver or the provider builtin answer".

### 4. V1 chat mappings are intentionally narrow

Start with only the direct mappings that are clearly justified:

| `models.dev` field                 | `ChatCapability` variant | V1 decision |
|-----------------------------------|--------------------------|-------------|
| `tool_call`                       | `ToolCalls`              | yes         |
| `structured_output`               | `StructuredOutput`       | yes         |
| `modalities.input` has `"image"` | `ImageInput`             | yes         |
| `modalities.output` has `"image"`| `ImageOutput`            | yes         |
| `reasoning`                       | `ReasoningOutput`        | no, return `None` in v1 |

Everything else returns `None` in v1, including:

- `Streaming`
- `NativeStreaming`
- `ParallelToolCalls`
- `ImageDetail`
- `ImageReplay`
- `ReasoningOutput`
- `ReasoningReplay`
- `ReasoningConfig`

The `reasoning` field should not be mapped in v1. Its semantics are not
yet clearly aligned with `ChatCapability::ReasoningOutput`, which refers
specifically to reasoning blocks in assistant responses.

### 5. Do not install the resolver by default in provider crates

Provider crates must not change their default capability behavior.

The `models.dev` resolver is an opt-in user feature, not a provider-side
default fallback layer.

Provider crates may offer a helper for constructing a resolver from an
embedded snapshot, but users choose whether to install it.

### 6. Provider crates may ship checked-in per-provider snapshots

Each provider crate that opts in may ship a checked-in JSON file
containing only its provider entry from `models.dev`.

Examples:

```text
crates/anyllm-openai/models-dev.json
crates/anyllm-anthropic/models-dev.json
crates/anyllm-gemini/models-dev.json
```

These files are:

- committed to git
- inspectable and diffable
- embedded at compile time via `include_str!()`
- updated manually with a simple script

The snapshot may contain non-chat models as well, but v1 resolver logic
only applies to chat capability queries.

### 7. Expose an opt-in helper from provider crates

Provider crates that enable a `models-dev` feature may expose a helper
that constructs the resolver from the embedded snapshot.

Example shape:

```rust
pub fn models_dev_chat_capabilities() -> serde_json::Result<ModelsDevResolver> {
    ModelsDevResolver::from_provider_json(include_str!("../models-dev.json"))
}
```

Exact naming is less important than the behavior: users should not have to
manually locate snapshot files or deserialize JSON to opt in.

### 8. Users opt in explicitly through `with_chat_capabilities(...)`

Example:

```rust
let provider = anyllm_openai::Provider::new(api_key)?
    .with_chat_capabilities(anyllm_openai::models_dev_chat_capabilities()?);
```

### 9. Add ordered resolver composition in `anyllm` core

`anyllm` should support composing multiple capability resolvers without
forcing callers to hand-roll a wrapper type.

For chat capability resolvers, add implementations for:

- `Arc<T>`
- `Vec<T>`

(`Box<T>` is omitted because `std` implements `Fn` for `Box<F>`, which
creates a coherence conflict with the closure blanket impl.)

with ordered semantics:

- resolvers are consulted in order
- the first `Some(...)` wins
- if all resolvers return `None`, provider builtin logic runs next

This allows user-controlled layering such as:

1. local application overrides
2. `models.dev` advisory answers
3. provider builtin fallback

Example use:

```rust
use std::sync::Arc;

let provider = anyllm_openai::Provider::new(api_key)?.with_chat_capabilities(vec![
    Arc::new(my_custom_resolver) as Arc<dyn ChatCapabilityResolver>,
    Arc::new(anyllm_openai::models_dev_chat_capabilities()?),
]);
```

### 10. Add embedding capability resolver parity in `anyllm` core

For parity with chat capability composition, add an
`EmbeddingCapabilityResolver` trait to `anyllm` core with the same
defer/override semantics:

- `Some(...)` overrides
- `None` defers

Add matching implementations for:

- `Arc<T>`
- `Vec<T>`

(Same `Box<T>` omission as chat, for the same coherence reason.)

Embedding-capable providers should gain a corresponding opt-in
`with_embedding_capabilities(...)` hook that mirrors
`with_chat_capabilities(...)`.

This issue does **not** add a `models.dev`-backed embedding resolver in v1.
The embedding resolver work here is limited to core parity and user-facing
composition semantics.

## Extraction workflow

A simple script is sufficient for v1.

Example:

```sh
#!/bin/sh
set -eu

API=$(curl -fsSL https://models.dev/api.json)

printf '%s' "$API" | jq '.openai'    > crates/anyllm-openai/models-dev.json
printf '%s' "$API" | jq '.anthropic' > crates/anyllm-anthropic/models-dev.json
printf '%s' "$API" | jq '.google'    > crates/anyllm-gemini/models-dev.json
```

Snapshots are updated manually and committed to git.

## Required changes

### `anyllm-models`

- add a `resolver` feature that depends on `anyllm`
- add `ModelsDevResolver`
- add `from_provider_json`

### `anyllm`

- add `ChatCapabilityResolver` implementations for `Box<T>`, `Arc<T>`, and `Vec<T>`
- add `EmbeddingCapabilityResolver` with the same `Option<CapabilitySupport>` semantics
- add `EmbeddingCapabilityResolver` implementations for `Box<T>`, `Arc<T>`, and `Vec<T>`

### provider crates

- optionally add a `models-dev` feature gating a dependency on `anyllm-models`
- optionally add a checked-in `models-dev.json` snapshot
- expose an opt-in helper for building a chat resolver from the embedded snapshot
- add `with_embedding_capabilities(...)` where the provider supports embeddings

## Benefits

- fills some `Unknown` chat capability gaps without changing default provider behavior
- keeps `models.dev` integration outside `anyllm` core
- gives users explicit control over whether advisory snapshot data is used
- makes resolver composition ergonomic instead of one-off
- establishes consistent composition semantics across chat and embeddings

## Scope boundaries

Non-goals for v1:

- default installation of the `models.dev` resolver in provider crates
- runtime fetching of `models.dev` data for capability resolution
- automatic snapshot refresh or CI-driven updates
- inferring capability from model names or other heuristics
- mapping ambiguous `models.dev` fields to portable capabilities
- using `models.dev` for embedding capability answers in v1
- using `models.dev` for routing, cost-aware selection, or token-limit validation

## Remaining questions

- which provider crate should be the first opt-in integration target?
- what is the best helper API surface for provider crates?
  - free function
  - associated constructor
  - both
- should provider crates expose snapshot parse failure directly as
  `serde_json::Error`, or wrap it in a provider-specific error type?

## Suggested acceptance criteria

- `anyllm-models` has an opt-in `resolver` feature depending on `anyllm`
- `anyllm-models` exposes `ModelsDevResolver`
- `ModelsDevResolver` implements `ChatCapabilityResolver`
- the resolver returns `Some(...)` only for explicit, accepted mappings
- the resolver returns `None` for missing or semantically unclear data
- `reasoning` is not mapped to `ReasoningOutput` in v1
- `anyllm` implements ordered composition for `ChatCapabilityResolver` via
  `Box<T>`, `Arc<T>`, and `Vec<T>`
- `anyllm` adds `EmbeddingCapabilityResolver` with matching composition
  support via `Box<T>`, `Arc<T>`, and `Vec<T>`
- at least one provider crate ships an opt-in `models-dev` feature and
  an embedded per-provider snapshot
- at least one provider crate exposes a helper for constructing a
  `models.dev` chat resolver from that snapshot
- provider default capability behavior is unchanged when users do not opt in
- tests cover:
  - mapped capability returns `Some(...)`
  - missing field returns `None`
  - unclear mapping returns `None`
  - ordered resolver composition for chat
  - ordered resolver composition for embeddings
  - provider builtin behavior remains unchanged without opt-in
  - embedded snapshot JSON deserializes successfully
