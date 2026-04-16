//! Cloudflare Workers AI provider for anyllm.
//!
//! This crate implements [`anyllm::ChatProvider`] using the `worker::Ai` binding,
//! enabling AI model calls directly from Cloudflare Workers without HTTP overhead.
//! Construction is runtime-bound: unlike the HTTP providers, this adapter is not
//! loaded from local environment variables and instead requires a live Worker
//! `Ai` binding.
//!
//! # Usage
//!
//! ```rust,ignore
//! use anyllm::prelude::*;
//! use anyllm_cloudflare_worker::Provider;
//!
//! async fn handler(env: &worker::Env) -> Result<String> {
//!     let ai = env.ai("AI").map_err(|e| anyllm::Error::Provider {
//!         status: None,
//!         message: e.to_string(),
//!         body: None,
//!         request_id: None,
//!     })?;
//!
//!     let provider = Provider::new(ai);
//!
//!     let response = provider.chat(
//!         &ChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
//!             .message(Message::user("What is Rust?"))
//!     ).await?;
//!
//!     Ok(response.text_or_empty())
//! }
//! ```
//!
//! # Streaming
//!
//! Streaming uses `Ai::run_bytes()` with `stream: true`, parsing SSE events
//! from the raw byte stream:
//!
//! ```rust,ignore
//! use anyllm::prelude::*;
//! use anyllm_cloudflare_worker::Provider;
//! use futures_util::StreamExt;
//!
//! async fn stream_handler(env: &worker::Env) -> Result<()> {
//!     let ai = env.ai("AI").map_err(|e| anyllm::Error::Provider {
//!         status: None,
//!         message: e.to_string(),
//!         body: None,
//!         request_id: None,
//!     })?;
//!
//!     let provider = Provider::new(ai);
//!
//!     let mut stream = provider.chat_stream(
//!         &ChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
//!             .message(Message::user("Tell me a story"))
//!     ).await?;
//!
//!     while let Some(event) = stream.next().await {
//!         let event = event?;
//!         if let StreamEvent::TextDelta { text, .. } = &event {
//!             // Process streaming text
//!         }
//!     }
//!     Ok(())
//! }
//! ```

mod chat;
#[cfg(test)]
mod conformance_tests;
mod embedding;
mod error;
mod streaming;
mod wire;

use std::sync::Arc;

use anyllm::{CapabilitySupport, ChatCapability, ChatCapabilityResolver};

/// Cloudflare Workers AI provider implementing `anyllm::ChatProvider`.
///
/// Wraps a `worker::Ai` binding to call AI models directly from within
/// a Cloudflare Worker. No HTTP overhead — calls go through the Workers
/// runtime's internal AI binding.
///
/// # Construction
///
/// Obtain the `worker::Ai` binding from the Worker's `Env`:
///
/// ```rust,ignore
/// let ai = env.ai("AI")?;
/// let provider = Provider::new(ai);
/// ```
///
/// The `"AI"` string must match the binding name in `wrangler.toml`:
///
/// ```toml
/// [ai]
/// binding = "AI"
/// ```
pub struct Provider {
    pub(crate) ai: worker::Ai,
    pub(crate) chat_capability_resolver: Option<Arc<dyn ChatCapabilityResolver>>,
}

impl Provider {
    /// Create a new provider from a `worker::Ai` binding.
    pub fn new(ai: worker::Ai) -> Self {
        Self {
            ai,
            chat_capability_resolver: None,
        }
    }

    fn builtin_chat_capability(
        &self,
        _model: &str,
        capability: ChatCapability,
    ) -> CapabilitySupport {
        match capability {
            ChatCapability::Streaming
            | ChatCapability::NativeStreaming
            | ChatCapability::ToolCalls
            | ChatCapability::StructuredOutput => CapabilitySupport::Supported,
            ChatCapability::ParallelToolCalls
            | ChatCapability::ImageInput
            | ChatCapability::ImageDetail
            | ChatCapability::ImageOutput
            | ChatCapability::ImageReplay
            | ChatCapability::ReasoningOutput
            | ChatCapability::ReasoningReplay
            | ChatCapability::ReasoningConfig => CapabilitySupport::Unsupported,
            _ => CapabilitySupport::Unknown,
        }
    }

    /// Install a resolver consulted before the provider's built-in capability logic.
    #[must_use]
    pub fn with_chat_capabilities(mut self, resolver: impl ChatCapabilityResolver) -> Self {
        self.chat_capability_resolver = Some(Arc::new(resolver));
        self
    }
}
