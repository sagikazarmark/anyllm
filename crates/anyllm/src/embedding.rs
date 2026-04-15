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

use crate::{RequestOptions, ResponseMetadata, Usage};

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

/// A provider-agnostic embedding response.
///
/// Embeddings are returned in the same order as `EmbeddingRequest.inputs`.
/// Provider-specific response details belong in [`ResponseMetadata`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct EmbeddingResponse {
    /// Ordered embedding vectors, one per input.
    pub embeddings: Vec<Vec<f32>>,
    /// Provider-reported model identifier when available.
    pub model: Option<String>,
    /// Token-usage information when the provider reports it.
    pub usage: Option<Usage>,
    /// Typed and portable provider-specific metadata.
    pub metadata: ResponseMetadata,
}

impl EmbeddingResponse {
    /// Create an embedding response from ordered vectors.
    #[must_use]
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            embeddings,
            model: None,
            usage: None,
            metadata: ResponseMetadata::new(),
        }
    }

    /// Set the reported model identifier.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the usage statistics for this response.
    #[must_use]
    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Replace the attached metadata.
    #[must_use]
    pub fn metadata(mut self, metadata: ResponseMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Build a portable JSON log value, skipping typed metadata that cannot
    /// be serialized. Matches the shape used by
    /// `anyllm-conformance::assert_embedding_response_fixture_eq`.
    #[must_use]
    pub fn to_log_value(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert(
            "embeddings".into(),
            serde_json::to_value(&self.embeddings)
                .expect("Vec<Vec<f32>> serialization should be infallible"),
        );
        if let Some(model) = &self.model {
            map.insert("model".into(), serde_json::Value::String(model.clone()));
        }
        if let Some(usage) = &self.usage {
            map.insert(
                "usage".into(),
                serde_json::to_value(usage).expect("Usage serialization should be infallible"),
            );
        }
        let metadata = self.metadata.to_portable_map();
        if !metadata.is_empty() {
            map.insert("metadata".into(), serde_json::Value::Object(metadata));
        }
        serde_json::Value::Object(map)
    }
}

/// Portable embedding features that a provider/model may support.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingCapability {
    /// Accepts more than one input in a single request.
    BatchInput,
    /// Honors the [`EmbeddingRequest::dimensions`] output-size request field.
    OutputDimensions,
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
        assert_eq!(
            request.inputs,
            vec!["hello".to_string(), "world".to_string()]
        );
        assert_eq!(request.dimensions, None);
    }

    #[test]
    fn inputs_replaces_existing() {
        let request = EmbeddingRequest::new("model").input("a").inputs(["b", "c"]);

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
        let request = EmbeddingRequest::new("model").with_option(ProviderHint { tag: "retrieval" });

        assert_eq!(
            request.option::<ProviderHint>(),
            Some(&ProviderHint { tag: "retrieval" })
        );
    }
}

#[cfg(test)]
mod response_tests {
    use super::*;

    #[test]
    fn response_builder_sets_fields() {
        let response = EmbeddingResponse::new(vec![vec![0.1, 0.2], vec![0.3, 0.4]])
            .model("text-embedding-3-small")
            .usage(Usage::new().input_tokens(10));

        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(response.model.as_deref(), Some("text-embedding-3-small"));
        assert_eq!(
            response.usage.as_ref().and_then(|u| u.input_tokens),
            Some(10)
        );
    }

    #[test]
    fn to_log_value_serializes_portable_shape() {
        let response = EmbeddingResponse::new(vec![vec![0.0, 1.0]])
            .model("text-embedding-3-small")
            .usage(Usage::new().input_tokens(5));

        let value = response.to_log_value();
        assert_eq!(
            value,
            serde_json::json!({
                "embeddings": [[0.0, 1.0]],
                "model": "text-embedding-3-small",
                "usage": { "input_tokens": 5 }
            })
        );
    }

    #[test]
    fn capability_is_copy_and_hashable() {
        use std::collections::HashSet;
        let mut set: HashSet<EmbeddingCapability> = HashSet::new();
        set.insert(EmbeddingCapability::BatchInput);
        set.insert(EmbeddingCapability::OutputDimensions);
        assert_eq!(set.len(), 2);
    }
}
