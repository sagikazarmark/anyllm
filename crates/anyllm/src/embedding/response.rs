use crate::{ResponseMetadata, Usage};

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

#[cfg(test)]
mod response_tests {
    use super::*;
    use crate::EmbeddingCapability;

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
