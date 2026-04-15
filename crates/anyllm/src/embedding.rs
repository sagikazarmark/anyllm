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

use crate::RequestOptions;

/// A provider-agnostic embedding request.
///
/// Text-only and batch-oriented. Single-input callers should prefer
/// [`crate::EmbeddingProviderExt::embed_text`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EmbeddingRequest {
    /// Provider-specific model identifier to route this request to.
    pub model: String,
    /// Ordered inputs to embed. A single-element vector is a valid one-shot
    /// request; multi-element vectors use the provider's batch API.
    pub inputs: Vec<String>,
    /// Optional requested output dimensionality. Providers that support this
    /// feature will produce vectors of this size; providers that do not should
    /// reject the request or report
    /// [`crate::CapabilitySupport::Unsupported`] through
    /// [`crate::EmbeddingProvider::embedding_capability`].
    pub dimensions: Option<u32>,
    /// Provider-specific typed request extensions.
    pub options: RequestOptions,
}

impl EmbeddingRequest {
    /// Create a new embedding request targeting the given model with no inputs.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            inputs: Vec::new(),
            dimensions: None,
            options: RequestOptions::new(),
        }
    }

    /// Set the inputs, replacing any existing values.
    #[must_use]
    pub fn inputs<I, S>(mut self, inputs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inputs = inputs.into_iter().map(Into::into).collect();
        self
    }

    /// Append a single input to the request.
    #[must_use]
    pub fn input(mut self, input: impl Into<String>) -> Self {
        self.inputs.push(input.into());
        self
    }

    /// Set the requested output dimensionality.
    #[must_use]
    pub fn dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Insert a typed provider-specific request option.
    #[must_use]
    pub fn with_option<T>(mut self, option: T) -> Self
    where
        T: Clone + Send + Sync + 'static,
    {
        self.options.insert(option);
        self
    }

    /// Borrow a typed provider-specific request option by type.
    pub fn option<T>(&self) -> Option<&T>
    where
        T: Send + Sync + 'static,
    {
        self.options.get::<T>()
    }
}

#[cfg(test)]
mod request_tests {
    use super::*;

    #[test]
    fn builder_sets_model_and_inputs() {
        let request = EmbeddingRequest::new("text-embedding-3-small")
            .input("hello")
            .input("world");

        assert_eq!(request.model, "text-embedding-3-small");
        assert_eq!(request.inputs, vec!["hello".to_string(), "world".to_string()]);
        assert_eq!(request.dimensions, None);
    }

    #[test]
    fn inputs_replaces_existing() {
        let request = EmbeddingRequest::new("model")
            .input("a")
            .inputs(["b", "c"]);

        assert_eq!(request.inputs, vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn dimensions_setter_stores_value() {
        let request = EmbeddingRequest::new("model").dimensions(512);
        assert_eq!(request.dimensions, Some(512));
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ProviderHint {
        tag: &'static str,
    }

    #[test]
    fn typed_options_round_trip() {
        let request =
            EmbeddingRequest::new("model").with_option(ProviderHint { tag: "retrieval" });

        assert_eq!(
            request.option::<ProviderHint>(),
            Some(&ProviderHint { tag: "retrieval" })
        );
    }
}
