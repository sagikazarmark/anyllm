#![warn(missing_docs)]
//! Provider-agnostic chat completion primitives and adapters.
//!
//! The crate root exposes the common day-to-day API surface directly:
//! [`ChatProvider`], [`ChatRequest`], [`ChatResponse`], [`Tool`], and related
//! streaming and wrapper types.
//!
//! Use [`prelude`] when you want a compact import for typical application code.
//! Reach for explicit root imports when writing libraries or examples that need
//! to make advanced types like [`ChatRequestRecord`], [`ChatResponseRecord`],
//! [`CollectedResponse`], or [`StreamCompleteness`] obvious at the callsite.
//!
//! ## Portability Model
//!
//! `anyllm` aims to keep the common request/response path provider-agnostic,
//! while still leaving explicit escape hatches for provider-specific features.
//!
//! The portable core is centered on [`ChatRequest`], [`ChatResponse`],
//! [`Message`], [`ContentBlock`], [`Tool`], and the streaming event model.
//! These types intentionally expose their portable fields directly and pair
//! them with fluent constructors/helpers so application and test code can build
//! and adjust requests and responses without going through opaque builders.
//!
//! Provider-specific data lives in a few deliberate places:
//! [`RequestOptions`] and [`ResponseMetadata`] for typed extensions,
//! [`ExtraMap`] and `extensions` fields for portable JSON-shaped extras, and
//! `Other` enum variants for provider payloads that do not fit the normalized
//! model yet.
//!
//! This means portability is best-effort rather than absolute. Converting to
//! [`ChatRequestRecord`] or [`ChatResponseRecord`] preserves the portable core,
//! but typed provider-specific data may be dropped or flattened to JSON.

mod capability;
mod chat;
mod embedding;
mod error;
mod identity;
mod options;
mod usage;
mod utils;

pub use capability::CapabilitySupport;
pub use chat::{
    AssistantMessageRef, ChatCapability, ChatCapabilityResolver, ChatProvider, ChatProviderExt,
    ChatRequest, ChatRequestRecord, ChatResponse, ChatResponseRecord, ChatStream, ChatStreamExt,
    CollectedResponse, ContentBlock, ContentPart, DynChatProvider, FallbackChatProvider,
    FinishReason, ImageBlockRef, ImagePartRef, ImageSource, Message, OwnedToolCall,
    ReasoningConfig, ReasoningEffort, ResponseFormat, RetryPolicy, RetryingChatProvider,
    SingleResponseStream, StreamBlockType, StreamCollector, StreamCompleteness, StreamEvent, Tool,
    ToolCallRef, ToolChoice, ToolMessageRef, ToolResultContent, UsageMetadataMode, UserContent,
    UserMessageRef,
};
#[cfg(any(test, feature = "mock"))]
pub use chat::{
    ChatResponseBuilder, MockProvider, MockProviderBuilder, MockResponse, MockStreamEvent,
    MockStreamingProvider, MockStreamingProviderBuilder, MockToolRoundTrip,
};
#[cfg(feature = "extract")]
pub use chat::{
    ExtractError, ExtractExt, Extracted, ExtractingProvider, ExtractionMetadata, ExtractionMode,
    ExtractionRequest, Extractor,
};
#[cfg(feature = "tracing")]
pub use chat::{TracingChatProvider, TracingContentConfig, otel_genai_provider_name};
#[cfg(any(test, feature = "mock"))]
pub use embedding::MockEmbeddingProvider;
pub use embedding::{
    DynEmbeddingProvider, EmbeddingCapability, EmbeddingProvider, EmbeddingProviderExt,
    EmbeddingRequest, EmbeddingResponse,
};
pub use error::{Error, ErrorLog, Result, SerializationError};
pub use identity::ProviderIdentity;
pub use options::{RequestOptions, ResponseMetadata, ResponseMetadataType};
pub use usage::Usage;

/// Portable JSON object used for provider-specific escape hatches.
/// Uses `Map<String, Value>` instead of `Value` to enforce object semantics at compile time.
pub type ExtraMap = serde_json::Map<String, serde_json::Value>;

/// Prelude module — import `use anyllm::prelude::*` for common application-facing types.
///
/// The prelude intentionally omits some more specialized record and stream
/// diagnostics types so those remain explicit when used.
pub mod prelude {
    pub use futures_util::StreamExt;

    pub use crate::{
        CapabilitySupport, ChatCapability, ChatCapabilityResolver, ChatProvider, ChatProviderExt,
        ChatRequest, ChatResponse, ChatStream, ChatStreamExt, ContentBlock, ContentPart,
        DynChatProvider, DynEmbeddingProvider, EmbeddingCapability, EmbeddingProvider,
        EmbeddingProviderExt, EmbeddingRequest, EmbeddingResponse, Error, ErrorLog, ExtraMap,
        FallbackChatProvider, FinishReason, ImageSource, Message, OwnedToolCall, ProviderIdentity,
        ReasoningConfig, ReasoningEffort, ResponseFormat, Result, RetryPolicy,
        RetryingChatProvider, SingleResponseStream, StreamBlockType, StreamCollector, StreamEvent,
        Tool, ToolCallRef, ToolChoice, ToolResultContent, Usage, UserContent,
    };

    #[cfg(any(test, feature = "mock"))]
    pub use crate::{
        ChatResponseBuilder, MockEmbeddingProvider, MockProvider, MockProviderBuilder,
        MockResponse, MockStreamEvent, MockStreamingProvider, MockStreamingProviderBuilder,
        MockToolRoundTrip,
    };

    #[cfg(feature = "extract")]
    pub use crate::{
        ExtractError, ExtractExt, Extracted, ExtractingProvider, ExtractionMetadata,
        ExtractionMode, ExtractionRequest, Extractor,
    };

    #[cfg(feature = "tracing")]
    pub use crate::{TracingChatProvider, TracingContentConfig};
}
