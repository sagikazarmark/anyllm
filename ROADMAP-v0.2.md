# anyllm v0.2 Roadmap

Synthesized from an audit of the public surface, provider crates, and examples.
Short version: the core is in good shape — most v0.2 energy is better spent on
additive features than on breaks.

## Breaking changes worth making

Ordered high → low value. All are candidates, not musts.

### 1. Rethink the `prelude` + re-export surface — **high value, low cost**

The root `pub use` list re-exports ~40 types, and `prelude` re-exports ~30.
That makes `anyllm::` autocomplete noisy and forces everything into one public
namespace, which means future moves break callers. Examples already import both
`anyllm::ToolCallRef` *and* `anyllm::prelude::*`, which is telling — the
prelude doesn't actually cover the types you reach for.

- **Why high value**: this is the last chance to tighten the import surface
  before 1.0. It shapes what users type every day.
- **Direction**: trim the root `pub use` to top-level primitives; expose
  wrapper types (`RetryingChatProvider`, `FallbackChatProvider`,
  `TracingChatProvider`) under `anyllm::wrappers`, stream internals under
  `anyllm::stream`, extraction under `anyllm::extract`. Keep `prelude` focused
  on what an app actually calls (`ChatProvider`, `ChatRequest`,
  `ChatResponse`, `Message`, `Tool`, `StreamExt`). Make specialized types
  explicit.
- **Not**: changing semantics. Just paths.

### 2. `Tool.parameters: serde_json::Value` is a DX footgun — **high value**

Today tools are hand-authored JSON schemas inline (see
`crates/anyllm/examples/tool_calling.rs:13-24`). That's error-prone, duplicates
argument struct fields, and defeats schema validation at build time.
`schemars` is already a workspace dep behind the `extract` feature.

- **Direction**: keep `Tool::new(name, Value)` as the low-level constructor
  but add `Tool::from_schema::<T>(name)` (where `T: JsonSchema`) as the
  ergonomic path. Non-breaking at the type level, but worth pairing with
  tightening: either (a) change `parameters` to a lightweight schema newtype
  that derefs to `Value` and validates "must be object schema" on
  construction, or (b) leave the field alone and call it additive. Leaning
  (b) — the only "break" is you can no longer assume the field is untouched.
- Arguably this is **not** a break and belongs in the features section.
  Included here because a newtype *could* be worth it.

### 3. `ResponseFormat` / structured output surface — **medium value, verify first**

Worth auditing whether the current `ResponseFormat` cleanly covers: free text,
JSON mode, JSON schema (strict), and Anthropic-style "prefill to constrain."
If it doesn't, adding variants is non-breaking due to `#[non_exhaustive]`, but
changing the *shape* of the existing variants is breaking and should happen
now not later.

- **Action item**: read the enum before committing. If the variants model
  "what kind of response" cleanly, no break. If there's leaky provider shape
  (e.g. an OpenAI-style `json_schema` discriminator baked in), break it now.

### 4. Things **not** worth breaking

- **`SystemPrompt` + `SystemOptions`**: the type-erased options bag that skips
  serde looks weird at first but is the *designed* escape hatch for Anthropic
  cache-control hints, with a real motivating example
  (`anthropic_prompt_caching.rs`). It mirrors `RequestOptions` deliberately.
  Keep.
- **`*Ref` naming (`ToolCallRef`, `ImagePartRef`, `ToolMessageRef`)**:
  consistent convention for borrowed views of `ContentBlock`/`Message`
  variants. Leave it.
- **`RequestOptions` / `ResponseMetadata` dual typed+portable store**: pulling
  its weight; providers use it cleanly.
- **Streaming event model (`StreamEvent`, `StreamCollector`,
  `StreamCompleteness`)**: mature. No observable friction.
- **`ChatRequestRecord` / `ChatResponseRecord`**: explicit lossy conversion is
  the right shape for logging/fixtures. Keep.
- **`ExtraMap` / `extensions`**: right placement, not over-used.
- **`CapabilitySupport` / `ChatCapability` / `EmbeddingCapability`**:
  three-state query is correct. Keep.
- **`DynChatProvider` / `DynEmbeddingProvider`**: object-safe, clean.
- **`ProviderIdentity`**: open `&'static str`, correct choice for a lib that
  wants third-party providers.
- **Feature flags (`mock`, `extract`, `tracing`)**: right granularity.
- **Error model**: `RateLimited`/`Overloaded` already carry `retry_after`; the
  taxonomy is good. See "additions" for the one gap.

## New features worth adding (non-breaking)

Ordered high → low value.

### 1. Portable prompt-caching abstraction — **high value**

Anthropic, OpenAI (Responses API), and Gemini all ship server-side prompt
caching now. Today it lives as `anyllm_anthropic::CacheControl` attached via
`SystemPrompt::with_option(...)`. That's fine as an escape hatch but forces
per-provider code in callers.

- **Why high value**: this is the single largest cost-savings lever for real
  applications, and cross-provider users hit it immediately.
- **Why portable-worthy**: the semantics converge on "mark a cacheable
  boundary"; the knobs (TTL, tier) are small.
- **Direction**: a core `CacheControl` (or `CacheHint`) type that providers
  *may* honor, attached to `SystemPrompt` and `Message` via typed options.
  Providers that don't support it just ignore the hint. Report cache hits via
  `ResponseMetadata` + `Usage` (add `cached_input_tokens`).
- **Guardrail per CLAUDE.md**: design the neutral domain model first. Don't
  wrap Anthropic's struct.

### 2. `Usage` enrichment: cached / reasoning / audio tokens — **high value**

The `Usage` struct today almost certainly tracks input/output tokens. Every
major provider now separately reports cached-input tokens and reasoning tokens
(OpenAI `o*`, Anthropic extended thinking, Gemini thinking). Without this,
users can't tell cache hits or reasoning cost from a successful response.

- **Direction**: add `cached_input_tokens`, `reasoning_tokens`, and
  (separately) `audio_input_tokens`/`audio_output_tokens`. Non-exhaustive
  struct → additive.

### 3. Schemars-backed `Tool::from_schema::<T>()` — **high value**

See break #2 above. The cleanest, most useful form is additive: gate on the
existing `extract` feature (which already pulls schemars) or a new `schema`
feature.

### 4. Error metadata trait for uniform retry-after / request-id extraction — **medium value**

`Error::RateLimited { retry_after, .. }` and `Error::Overloaded {
retry_after, .. }` already carry this, but provider-specific error variants
inside `Error::Provider(..)` don't expose a uniform query path. Adds friction
to custom retry logic.

- **Direction**: small trait `ErrorInfo { fn retry_after(&self) ->
  Option<Duration>; fn request_id(&self) -> Option<&str>; }` implemented on
  `Error`. Additive.

### 5. Transcription / speech capability traits — **medium value, on demand**

`CLAUDE.md` explicitly puts these in scope. Don't design them until a provider
crate wants to land one, to avoid the "wrap the first provider" trap the doc
warns about. But flag it as an accepted direction so v0.2 doesn't paint
transcription into a corner.

### 6. Streaming `StreamCollector` + extraction composition polish — **low value**

Look for any awkwardness between `ChatStreamExt::collect_response()` and
`ExtractExt`. If streaming + extract requires buffering the whole stream
first, a `StreamingExtractor` helper could be added later — defer unless users
actually hit it.

## Open questions before committing

1. Does `ResponseFormat` today cleanly model strict JSON-schema + Anthropic
   prefill, or is there hidden provider shape? (Check before deciding break
   #3.)
2. Is there appetite to bump MSRV in v0.2? The workspace is on edition 2024 /
   Rust 1.92 already, so there's headroom but no obvious forcing function.
3. Provider-specific server-side tools (OpenAI web search, code interpreter;
   Anthropic computer use; Gemini search grounding) — out of scope for v0.2
   per CLAUDE.md, right? They're not portable yet. Confirm.

## Bottom line

- **Breaks**: 1 clearly worth it (prelude/re-export tightening), 1 conditional
  (`ResponseFormat` — verify first), 1 borderline (`Tool.parameters` newtype —
  probably better as additive). Everything else: leave alone.
- **Additions**: lead with **prompt caching + usage enrichment + schemars
  tools**. Those three alone would make v0.2 feel like a meaningful step
  forward for real users without touching the core shape.
