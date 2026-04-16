//! Provider-agnostic embedding primitives.
//!
//! This module defines the portable core for embedding operations: a batch
//! request with optional output dimensionality, a response with ordered vector
//! outputs, a small capability vocabulary, and a sibling trait to
//! [`crate::ChatProvider`].
//!
//! The portable surface is intentionally narrow. Provider-specific concerns
//! such as retrieval task hints, truncation policy, output encoding, and native
//! pooling should live in typed [`crate::RequestOptions`] entries rather than
//! in this shared API.

mod provider;
mod request;
mod response;

#[cfg(any(test, feature = "mock"))]
mod mock;

pub use provider::{
    DynEmbeddingProvider, EmbeddingCapability, EmbeddingCapabilityResolver, EmbeddingProvider,
    EmbeddingProviderExt,
};
pub use request::EmbeddingRequest;
pub use response::EmbeddingResponse;

#[cfg(any(test, feature = "mock"))]
pub use mock::MockEmbeddingProvider;
