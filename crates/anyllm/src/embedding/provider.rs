use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{CapabilitySupport, EmbeddingRequest, EmbeddingResponse, ProviderIdentity, Result};

/// Portable embedding features that a provider/model may support.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbeddingCapability {
    /// Accepts more than one input in a single request.
    BatchInput,
    /// Honors the [`EmbeddingRequest::dimensions`] output-size request field.
    OutputDimensions,
}

/// Optional resolver used to customize a provider's embedding capability answers.
///
/// Return `None` to defer to the provider's built-in answer. Return
/// `Some(...)` to override it, including `Some(CapabilitySupport::Unknown)`.
///
/// Composition is supported through [`Arc<dyn EmbeddingCapabilityResolver>`]
/// and [`Vec<T>`] impls. `Box<T>` is intentionally omitted because `std`
/// implements `Fn` for `Box<F>`, which creates a coherence conflict with the
/// closure blanket impl.
pub trait EmbeddingCapabilityResolver: Send + Sync + 'static {
    /// Return an override for the queried capability, or `None` to defer to
    /// the provider's built-in capability logic.
    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> Option<CapabilitySupport>;
}

impl<F> EmbeddingCapabilityResolver for F
where
    F: for<'a> Fn(&'a str, EmbeddingCapability) -> Option<CapabilitySupport>
        + Send
        + Sync
        + 'static,
{
    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> Option<CapabilitySupport> {
        self(model, capability)
    }
}

impl EmbeddingCapabilityResolver for Arc<dyn EmbeddingCapabilityResolver> {
    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> Option<CapabilitySupport> {
        (**self).embedding_capability(model, capability)
    }
}

impl<T: EmbeddingCapabilityResolver> EmbeddingCapabilityResolver for Vec<T> {
    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> Option<CapabilitySupport> {
        for resolver in self {
            if let Some(support) = resolver.embedding_capability(model, capability) {
                return Some(support);
            }
        }
        None
    }
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

/// Convenience extension methods for [`EmbeddingProvider`] implementors.
pub trait EmbeddingProviderExt: EmbeddingProvider {
    /// Quick one-shot embedding for a single input.
    ///
    /// # Errors
    ///
    /// Propagates any [`crate::Error`] from the underlying
    /// [`EmbeddingProvider::embed`] call, and returns
    /// [`crate::Error::UnexpectedResponse`] if the provider response contains
    /// no embeddings.
    fn embed_text(
        &self,
        model: &str,
        input: impl Into<String>,
    ) -> impl Future<Output = Result<Vec<f32>>> + Send {
        let input = input.into();
        let model = model.to_string();

        async move {
            let response = self
                .embed(&EmbeddingRequest::new(model).input(input))
                .await?;

            response.embeddings.into_iter().next().ok_or_else(|| {
                crate::Error::UnexpectedResponse(format!(
                    "provider '{}' returned no embeddings for embed_text()",
                    self.provider_name()
                ))
            })
        }
    }
}

impl<T: EmbeddingProvider> EmbeddingProviderExt for T {}

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

#[cfg(test)]
mod resolver_tests {
    use super::*;
    use crate::CapabilitySupport;
    use std::sync::Arc;

    struct FixedEmbeddingResolver(Option<CapabilitySupport>);

    impl EmbeddingCapabilityResolver for FixedEmbeddingResolver {
        fn embedding_capability(
            &self,
            _model: &str,
            _capability: EmbeddingCapability,
        ) -> Option<CapabilitySupport> {
            self.0
        }
    }

    #[test]
    fn closure_resolver_works() {
        let resolver = |_model: &str, _cap: EmbeddingCapability| -> Option<CapabilitySupport> {
            Some(CapabilitySupport::Supported)
        };
        assert_eq!(
            resolver.embedding_capability("m", EmbeddingCapability::BatchInput),
            Some(CapabilitySupport::Supported),
        );
    }

    #[test]
    fn arc_delegates_to_inner() {
        let resolver: Arc<dyn EmbeddingCapabilityResolver> =
            Arc::new(FixedEmbeddingResolver(Some(CapabilitySupport::Unsupported)));
        assert_eq!(
            resolver.embedding_capability("m", EmbeddingCapability::BatchInput),
            Some(CapabilitySupport::Unsupported),
        );
    }

    #[test]
    fn vec_returns_first_some() {
        let resolvers: Vec<Arc<dyn EmbeddingCapabilityResolver>> = vec![
            Arc::new(FixedEmbeddingResolver(None)),
            Arc::new(FixedEmbeddingResolver(Some(CapabilitySupport::Supported))),
            Arc::new(FixedEmbeddingResolver(Some(CapabilitySupport::Unsupported))),
        ];
        assert_eq!(
            resolvers.embedding_capability("m", EmbeddingCapability::BatchInput),
            Some(CapabilitySupport::Supported),
        );
    }

    #[test]
    fn vec_returns_none_when_all_defer() {
        let resolvers: Vec<Arc<dyn EmbeddingCapabilityResolver>> = vec![
            Arc::new(FixedEmbeddingResolver(None)),
            Arc::new(FixedEmbeddingResolver(None)),
        ];
        assert_eq!(
            resolvers.embedding_capability("m", EmbeddingCapability::BatchInput),
            None,
        );
    }

    #[test]
    fn empty_vec_returns_none() {
        let resolvers: Vec<Arc<dyn EmbeddingCapabilityResolver>> = vec![];
        assert_eq!(
            resolvers.embedding_capability("m", EmbeddingCapability::BatchInput),
            None,
        );
    }
}

#[cfg(test)]
mod ext_tests {
    use super::*;
    use crate::{Error, ProviderIdentity};
    use std::sync::Mutex;

    struct RecordingProvider {
        response: EmbeddingResponse,
        last_inputs: Mutex<Option<Vec<String>>>,
    }

    impl ProviderIdentity for RecordingProvider {
        fn provider_name(&self) -> &'static str {
            "recording"
        }
    }

    impl EmbeddingProvider for RecordingProvider {
        async fn embed(&self, request: &EmbeddingRequest) -> crate::Result<EmbeddingResponse> {
            *self.last_inputs.lock().unwrap() = Some(request.inputs.clone());
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn embed_text_sends_single_input_and_returns_vector() {
        let provider = RecordingProvider {
            response: EmbeddingResponse::new(vec![vec![0.5, 0.5]]),
            last_inputs: Mutex::new(None),
        };
        let vector = provider.embed_text("model", "hello").await.unwrap();
        assert_eq!(vector, vec![0.5, 0.5]);
        assert_eq!(
            provider.last_inputs.lock().unwrap().clone(),
            Some(vec!["hello".to_string()])
        );
    }

    #[tokio::test]
    async fn embed_text_errors_when_response_has_no_vectors() {
        let provider = RecordingProvider {
            response: EmbeddingResponse::new(Vec::new()),
            last_inputs: Mutex::new(None),
        };
        let err = provider.embed_text("model", "hello").await.unwrap_err();
        match err {
            Error::UnexpectedResponse(message) => assert!(
                message.contains("recording"),
                "expected provider name in error, got: {message}"
            ),
            other => panic!("expected UnexpectedResponse, got {other:?}"),
        }
    }
}
