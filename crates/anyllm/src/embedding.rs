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

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::{CapabilitySupport, ProviderIdentity, RequestOptions, ResponseMetadata, Result, Usage};

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

/// Core trait for providers that expose a text embedding API.
///
/// Implementations are batch-oriented. Callers that have a single input
/// should use [`EmbeddingProviderExt::embed_text`].
///
/// Methods return `impl Future<…> + Send` so wrappers and dyn dispatch can
/// rely on `Send` futures, matching [`crate::ChatProvider`].
pub trait EmbeddingProvider: ProviderIdentity {
    /// Send an embedding request and return ordered vectors.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error`] on provider communication or decoding failures.
    fn embed(
        &self,
        request: &EmbeddingRequest,
    ) -> impl Future<Output = Result<EmbeddingResponse>> + Send;

    /// Returns support information for a provider/model embedding capability.
    fn embedding_capability(
        &self,
        _model: &str,
        _capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        CapabilitySupport::Unknown
    }
}

impl<T> EmbeddingProvider for &T
where
    T: EmbeddingProvider + ?Sized,
{
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        T::embed(*self, request).await
    }

    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        T::embedding_capability(*self, model, capability)
    }
}

impl<T> EmbeddingProvider for Box<T>
where
    T: EmbeddingProvider + ?Sized,
{
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        T::embed(self.as_ref(), request).await
    }

    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        T::embedding_capability(self.as_ref(), model, capability)
    }
}

impl<T> EmbeddingProvider for Arc<T>
where
    T: EmbeddingProvider + ?Sized,
{
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        T::embed(self.as_ref(), request).await
    }

    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        T::embedding_capability(self.as_ref(), model, capability)
    }
}

/// A type-erased embedding provider for dynamic dispatch.
///
/// Wraps any `T: EmbeddingProvider + 'static` behind a vtable, boxing the
/// async method future. Mirrors [`crate::DynChatProvider`].
#[derive(Clone)]
pub struct DynEmbeddingProvider(Arc<dyn EmbeddingProviderErased>);

impl DynEmbeddingProvider {
    /// Erase a concrete provider into a `DynEmbeddingProvider`.
    #[must_use]
    pub fn new<T>(provider: T) -> Self
    where
        T: EmbeddingProvider + 'static,
    {
        Self(Arc::new(provider))
    }
}

impl<T> From<Arc<T>> for DynEmbeddingProvider
where
    T: EmbeddingProvider + 'static,
{
    fn from(provider: Arc<T>) -> Self {
        Self(provider)
    }
}

impl std::fmt::Debug for DynEmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynEmbeddingProvider")
            .field("provider", &self.0.provider_name())
            .finish()
    }
}

impl ProviderIdentity for DynEmbeddingProvider {
    fn provider_name(&self) -> &'static str {
        self.0.provider_name()
    }
}

impl EmbeddingProvider for DynEmbeddingProvider {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.0.embed_erased(request).await
    }

    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        self.0.embedding_capability_erased(model, capability)
    }
}

/// Object-safe internal trait that manually boxes the async `embed` future.
///
/// Sealed by the blanket impl for `T: EmbeddingProvider`.
trait EmbeddingProviderErased: ProviderIdentity {
    fn embed_erased<'a>(
        &'a self,
        request: &'a EmbeddingRequest,
    ) -> Pin<Box<dyn Future<Output = Result<EmbeddingResponse>> + Send + 'a>>;

    fn embedding_capability_erased(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport;
}

impl<T> EmbeddingProviderErased for T
where
    T: EmbeddingProvider,
{
    fn embed_erased<'a>(
        &'a self,
        request: &'a EmbeddingRequest,
    ) -> Pin<Box<dyn Future<Output = Result<EmbeddingResponse>> + Send + 'a>> {
        Box::pin(EmbeddingProvider::embed(self, request))
    }

    fn embedding_capability_erased(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        EmbeddingProvider::embedding_capability(self, model, capability)
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

#[cfg(test)]
mod provider_tests {
    use super::*;
    use crate::{ProviderIdentity, Result};
    use std::sync::Arc;

    struct StaticEmbeddingProvider {
        response: EmbeddingResponse,
    }

    impl ProviderIdentity for StaticEmbeddingProvider {
        fn provider_name(&self) -> &'static str {
            "static-embed"
        }
    }

    impl EmbeddingProvider for StaticEmbeddingProvider {
        async fn embed(&self, _request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
            Ok(self.response.clone())
        }

        fn embedding_capability(
            &self,
            _model: &str,
            capability: EmbeddingCapability,
        ) -> CapabilitySupport {
            match capability {
                EmbeddingCapability::BatchInput => CapabilitySupport::Supported,
                EmbeddingCapability::OutputDimensions => CapabilitySupport::Unsupported,
            }
        }
    }

    fn demo_provider() -> StaticEmbeddingProvider {
        StaticEmbeddingProvider {
            response: EmbeddingResponse::new(vec![vec![0.1, 0.2]]).model("demo"),
        }
    }

    #[tokio::test]
    async fn direct_impl_returns_response() {
        let provider = demo_provider();
        let request = EmbeddingRequest::new("demo").input("hello");
        let response = provider.embed(&request).await.unwrap();
        assert_eq!(response.embeddings, vec![vec![0.1, 0.2]]);
    }

    #[tokio::test]
    async fn ref_forwards_embed() {
        let provider = demo_provider();
        let borrowed: &StaticEmbeddingProvider = &provider;
        let request = EmbeddingRequest::new("demo").input("hello");
        assert_eq!(
            borrowed.embed(&request).await.unwrap().embeddings,
            vec![vec![0.1, 0.2]]
        );
        assert_eq!(borrowed.provider_name(), "static-embed");
    }

    #[tokio::test]
    async fn box_forwards_embed() {
        let boxed: Box<StaticEmbeddingProvider> = Box::new(demo_provider());
        let request = EmbeddingRequest::new("demo").input("hello");
        assert_eq!(
            boxed.embed(&request).await.unwrap().embeddings,
            vec![vec![0.1, 0.2]]
        );
    }

    #[tokio::test]
    async fn arc_forwards_embed_and_capability() {
        let arced: Arc<StaticEmbeddingProvider> = Arc::new(demo_provider());
        let request = EmbeddingRequest::new("demo").input("hello");
        assert_eq!(
            arced.embed(&request).await.unwrap().embeddings,
            vec![vec![0.1, 0.2]]
        );
        assert_eq!(
            arced.embedding_capability("demo", EmbeddingCapability::BatchInput),
            CapabilitySupport::Supported
        );
        assert_eq!(
            arced.embedding_capability("demo", EmbeddingCapability::OutputDimensions),
            CapabilitySupport::Unsupported
        );
    }

    #[tokio::test]
    async fn default_capability_method_returns_unknown() {
        struct Minimal;
        impl ProviderIdentity for Minimal {}
        impl EmbeddingProvider for Minimal {
            async fn embed(&self, _request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
                Ok(EmbeddingResponse::default())
            }
        }

        assert_eq!(
            Minimal.embedding_capability("any", EmbeddingCapability::BatchInput),
            CapabilitySupport::Unknown
        );
    }
}

#[cfg(test)]
mod dyn_tests {
    use super::*;
    use crate::ProviderIdentity;
    use std::sync::Arc;

    struct DynDemo {
        tag: &'static str,
    }

    impl ProviderIdentity for DynDemo {
        fn provider_name(&self) -> &'static str {
            self.tag
        }
    }

    impl EmbeddingProvider for DynDemo {
        async fn embed(&self, request: &EmbeddingRequest) -> crate::Result<EmbeddingResponse> {
            let inputs = request.inputs.len();
            Ok(EmbeddingResponse::new(vec![vec![0.0; 4]; inputs]))
        }

        fn embedding_capability(
            &self,
            _model: &str,
            capability: EmbeddingCapability,
        ) -> CapabilitySupport {
            match capability {
                EmbeddingCapability::BatchInput => CapabilitySupport::Supported,
                EmbeddingCapability::OutputDimensions => CapabilitySupport::Unsupported,
            }
        }
    }

    #[tokio::test]
    async fn dyn_provider_from_concrete_forwards_calls() {
        let provider = DynEmbeddingProvider::new(DynDemo { tag: "dyn-embed" });
        let request = EmbeddingRequest::new("demo").inputs(["a", "b"]);
        let response = provider.embed(&request).await.unwrap();
        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(provider.provider_name(), "dyn-embed");
        assert_eq!(
            provider.embedding_capability("demo", EmbeddingCapability::BatchInput),
            CapabilitySupport::Supported
        );
    }

    #[tokio::test]
    async fn dyn_provider_from_arc_is_cloneable() {
        let provider: DynEmbeddingProvider = Arc::new(DynDemo { tag: "arc-embed" }).into();
        let cloned = provider.clone();
        let request = EmbeddingRequest::new("demo").input("x");
        assert_eq!(cloned.embed(&request).await.unwrap().embeddings.len(), 1);
        assert_eq!(cloned.provider_name(), "arc-embed");
    }

    #[test]
    fn dyn_provider_debug_includes_provider_name() {
        let provider = DynEmbeddingProvider::new(DynDemo { tag: "debug-embed" });
        let debug = format!("{provider:?}");
        assert!(debug.contains("DynEmbeddingProvider"));
        assert!(debug.contains("debug-embed"));
    }
}
