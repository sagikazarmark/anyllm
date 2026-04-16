use anyllm::{CapabilitySupport, ChatCapability, ChatCapabilityResolver, EmbeddingCapability, EmbeddingCapabilityResolver, Error, Result};
use std::{sync::Arc, time::Duration};

mod chat;
#[cfg(test)]
mod conformance_tests;
mod embedding;
mod error;
mod options;
mod streaming;
mod wire;

pub use options::{ChatRequestOptions, EmbeddingRequestOptions};

#[cfg(feature = "http-tracing")]
type HttpClient = reqwest_middleware::ClientWithMiddleware;
#[cfg(not(feature = "http-tracing"))]
type HttpClient = reqwest::Client;

/// Google Gemini API provider implementing `anyllm::ChatProvider`.
///
/// Clone is cheap — internals are wrapped in `Arc`.
#[derive(Clone)]
pub struct Provider {
    inner: Arc<Inner>,
}

pub(crate) struct Inner {
    pub(crate) client: HttpClient,
    pub(crate) request_timeout: Option<Duration>,
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) chat_capability_resolver: Option<Arc<dyn ChatCapabilityResolver>>,
    pub(crate) embedding_capability_resolver: Option<Arc<dyn EmbeddingCapabilityResolver>>,
}

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(120);

pub(crate) fn request_timeout(stream: bool, default_timeout: Option<Duration>) -> Option<Duration> {
    if stream { None } else { default_timeout }
}

fn normalize_base_url(url: impl Into<String>) -> String {
    url.into().trim().trim_end_matches('/').to_string()
}

fn required_env_var(name: &'static str) -> Result<String> {
    let value = std::env::var(name).map_err(|_| Error::Auth(format!("{name} not set")))?;
    if value.trim().is_empty() {
        return Err(Error::Auth(format!("{name} not set")));
    }
    Ok(value)
}

fn required_builder_value(name: &'static str, value: Option<String>) -> Result<String> {
    match value {
        Some(value) if !value.trim().is_empty() => Ok(value),
        _ => Err(Error::InvalidRequest(format!("{name} is required"))),
    }
}

fn builder_base_url(base_url: Option<String>) -> Result<String> {
    match base_url {
        Some(url) if url.trim().is_empty() => {
            Err(Error::InvalidRequest("base_url cannot be empty".into()))
        }
        Some(url) => Ok(normalize_base_url(url)),
        None => Ok(DEFAULT_BASE_URL.to_string()),
    }
}

impl Provider {
    fn default_http_client() -> HttpClient {
        let base = reqwest::Client::builder()
            .build()
            .expect("default Gemini reqwest client config should be valid");
        #[cfg(feature = "http-tracing")]
        {
            reqwest_middleware::ClientBuilder::new(base)
                .with(reqwest_tracing::TracingMiddleware::<
                    reqwest_tracing::SpanBackendWithUrl,
                >::new())
                .build()
        }
        #[cfg(not(feature = "http-tracing"))]
        {
            base
        }
    }

    fn plain_http_client(client: reqwest::Client) -> HttpClient {
        #[cfg(feature = "http-tracing")]
        {
            reqwest_middleware::ClientBuilder::new(client).build()
        }
        #[cfg(not(feature = "http-tracing"))]
        {
            client
        }
    }

    fn builtin_chat_capability(
        &self,
        _model: &str,
        capability: ChatCapability,
    ) -> CapabilitySupport {
        match capability {
            ChatCapability::ToolCalls
            | ChatCapability::Streaming
            | ChatCapability::NativeStreaming
            | ChatCapability::ImageInput
            | ChatCapability::ImageOutput
            | ChatCapability::StructuredOutput
            | ChatCapability::ReasoningOutput
            | ChatCapability::ReasoningConfig => CapabilitySupport::Supported,
            ChatCapability::ParallelToolCalls | ChatCapability::ImageReplay => {
                CapabilitySupport::Unknown
            }
            ChatCapability::ImageDetail | ChatCapability::ReasoningReplay => {
                CapabilitySupport::Unsupported
            }
            _ => CapabilitySupport::Unknown,
        }
    }

    fn builtin_embedding_capability(
        &self,
        _model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        match capability {
            EmbeddingCapability::BatchInput | EmbeddingCapability::OutputDimensions => {
                CapabilitySupport::Supported
            }
            _ => CapabilitySupport::Unknown,
        }
    }

    /// Create with just an API key. Uses default base URL.
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::new_with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Create with an API key and explicit base URL.
    pub fn new_with_base_url(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Result<Self> {
        let api_key = required_builder_value("api_key", Some(api_key.into()))?;
        let base_url = builder_base_url(Some(base_url.into()))?;

        Ok(Self {
            inner: Arc::new(Inner {
                client: Self::default_http_client(),
                request_timeout: Some(DEFAULT_HTTP_TIMEOUT),
                api_key,
                base_url,
                chat_capability_resolver: None,
                embedding_capability_resolver: None,
            }),
        })
    }

    /// Install a resolver consulted before the provider's built-in capability logic.
    #[must_use]
    pub fn with_chat_capabilities(self, resolver: impl ChatCapabilityResolver) -> Self {
        Self {
            inner: Arc::new(Inner {
                client: self.inner.client.clone(),
                request_timeout: self.inner.request_timeout,
                api_key: self.inner.api_key.clone(),
                base_url: self.inner.base_url.clone(),
                chat_capability_resolver: Some(Arc::new(resolver)),
                embedding_capability_resolver: self.inner.embedding_capability_resolver.clone(),
            }),
        }
    }

    /// Install a resolver consulted before the provider's built-in embedding capability logic.
    #[must_use]
    pub fn with_embedding_capabilities(self, resolver: impl EmbeddingCapabilityResolver) -> Self {
        Self {
            inner: Arc::new(Inner {
                client: self.inner.client.clone(),
                request_timeout: self.inner.request_timeout,
                api_key: self.inner.api_key.clone(),
                base_url: self.inner.base_url.clone(),
                chat_capability_resolver: self.inner.chat_capability_resolver.clone(),
                embedding_capability_resolver: Some(Arc::new(resolver)),
            }),
        }
    }

    /// Create from environment variables.
    ///
    /// Required: `GEMINI_API_KEY`
    /// Optional: `GEMINI_BASE_URL`
    pub fn from_env() -> Result<Self> {
        let api_key = required_env_var("GEMINI_API_KEY")?;
        let base_url = std::env::var("GEMINI_BASE_URL").ok();

        match base_url {
            Some(url) if !url.trim().is_empty() => Self::new_with_base_url(api_key, url),
            _ => Self::new(api_key),
        }
    }

    /// Builder for full configuration.
    #[must_use]
    pub fn builder() -> ProviderBuilder {
        ProviderBuilder {
            api_key: None,
            base_url: None,
            client: None,
            request_timeout: Some(DEFAULT_HTTP_TIMEOUT),
        }
    }
}

/// Builder for configuring a Gemini provider.
pub struct ProviderBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    client: Option<HttpClient>,
    request_timeout: Option<Duration>,
}

impl ProviderBuilder {
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    #[must_use]
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(Provider::plain_http_client(client));
        self.request_timeout = None;
        self
    }

    #[cfg(feature = "http-tracing")]
    #[must_use]
    pub fn client_with_middleware(
        mut self,
        client: reqwest_middleware::ClientWithMiddleware,
    ) -> Self {
        self.client = Some(client);
        self.request_timeout = None;
        self
    }

    pub fn build(self) -> Result<Provider> {
        let api_key = required_builder_value("api_key", self.api_key)?;
        Ok(Provider {
            inner: Arc::new(Inner {
                client: self.client.unwrap_or_else(Provider::default_http_client),
                request_timeout: self.request_timeout,
                api_key,
                base_url: builder_base_url(self.base_url)?,
                chat_capability_resolver: None,
                embedding_capability_resolver: None,
            }),
        })
    }
}

#[cfg(any(test, feature = "bench-internals"))]
#[doc(hidden)]
pub fn conformance_stream_from_sse_text(text: &str) -> anyllm::ChatStream {
    let normalized = if text.ends_with("\n\n") {
        text.to_string()
    } else {
        format!("{text}\n\n")
    };

    streaming::sse_to_stream(futures_util::stream::iter([Ok::<Vec<u8>, std::io::Error>(
        normalized.into_bytes(),
    )]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyllm::{ChatCapability, ChatProvider};
    use std::sync::{LazyLock, Mutex};

    static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    struct EnvVarGuard {
        name: &'static str,
        original: Option<String>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: Option<&str>) -> Self {
            let original = std::env::var(name).ok();

            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }

            Self { name, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.original {
                    Some(value) => std::env::set_var(self.name, value),
                    None => std::env::remove_var(self.name),
                }
            }
        }
    }

    #[test]
    fn builder_requires_api_key() {
        let result = Provider::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_empty_api_key() {
        let err = match Provider::builder().api_key("   ").build() {
            Ok(_) => panic!("expected empty API key to be rejected"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::InvalidRequest(message) if message == "api_key is required"));
    }

    #[test]
    fn builder_rejects_empty_base_url() {
        let err = match Provider::builder()
            .api_key("test-key")
            .base_url("   ")
            .build()
        {
            Ok(_) => panic!("expected empty base URL to be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(err, Error::InvalidRequest(message) if message == "base_url cannot be empty")
        );
    }

    #[test]
    fn from_env_rejects_empty_api_key() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _api_key = EnvVarGuard::set("GEMINI_API_KEY", Some("   "));
        let _base_url = EnvVarGuard::set("GEMINI_BASE_URL", None);

        let err = match Provider::from_env() {
            Ok(_) => panic!("expected empty API key to be rejected"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::Auth(message) if message == "GEMINI_API_KEY not set"));
    }

    #[test]
    fn chat_capability_resolver_overrides_builtin_answer() {
        let provider =
            Provider::new("test-key")
                .unwrap()
                .with_chat_capabilities(|model: &str, capability| {
                    if model == "gemini-2.5-pro" && capability == ChatCapability::ReasoningOutput {
                        Some(CapabilitySupport::Unknown)
                    } else {
                        None
                    }
                });

        assert_eq!(
            provider.chat_capability("gemini-2.5-pro", ChatCapability::ReasoningOutput),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            provider.chat_capability("gemini-2.5-flash", ChatCapability::ReasoningOutput),
            CapabilitySupport::Supported
        );
    }

    #[test]
    fn builtin_gemini_capabilities_report_native_streaming() {
        let provider = Provider::new("test-key").unwrap();
        assert_eq!(
            provider.chat_capability("gemini-2.0-flash", ChatCapability::NativeStreaming),
            CapabilitySupport::Supported
        );
    }

    #[test]
    fn later_chat_capability_resolver_replaces_earlier_one() {
        let provider = Provider::new("test-key")
            .unwrap()
            .with_chat_capabilities(|_: &str, capability| {
                if capability == ChatCapability::ToolCalls {
                    Some(CapabilitySupport::Unsupported)
                } else {
                    None
                }
            })
            .with_chat_capabilities(|_: &str, capability| {
                if capability == ChatCapability::ToolCalls {
                    Some(CapabilitySupport::Supported)
                } else {
                    None
                }
            });

        assert_eq!(
            provider.chat_capability("gemini-2.5-pro", ChatCapability::ToolCalls),
            CapabilitySupport::Supported
        );
    }

    #[test]
    fn builtin_capabilities_stay_conservative_for_replay_and_parallel_controls() {
        let provider = Provider::new("test-key").unwrap();

        assert_eq!(
            provider.chat_capability("gemini-2.5-pro", ChatCapability::ParallelToolCalls),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            provider.chat_capability("gemini-2.5-pro", ChatCapability::ImageReplay),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            provider.chat_capability("gemini-2.5-pro", ChatCapability::StructuredOutput),
            CapabilitySupport::Supported
        );
    }

    #[test]
    fn base_url_is_normalized_when_constructed() {
        let provider = Provider::new_with_base_url(
            "test-key",
            "https://generativelanguage.googleapis.com/v1beta///",
        )
        .unwrap();
        assert_eq!(
            provider.inner.base_url,
            "https://generativelanguage.googleapis.com/v1beta"
        );
    }

    #[test]
    fn new_rejects_empty_api_key() {
        let err = match Provider::new("   ") {
            Ok(_) => panic!("expected empty API key to be rejected"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::InvalidRequest(message) if message == "api_key is required"));
    }

    #[test]
    fn new_with_base_url_rejects_empty_base_url() {
        let err = match Provider::new_with_base_url("test-key", "   ") {
            Ok(_) => panic!("expected empty base URL to be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(err, Error::InvalidRequest(message) if message == "base_url cannot be empty")
        );
    }

    #[test]
    fn builder_normalizes_base_url() {
        let provider = Provider::builder()
            .api_key("test-key")
            .base_url("https://example.com/v1beta///")
            .build()
            .unwrap();

        assert_eq!(provider.inner.base_url, "https://example.com/v1beta");
    }

    #[test]
    fn default_timeout_is_only_applied_to_non_streaming_requests() {
        assert_eq!(
            request_timeout(false, Some(DEFAULT_HTTP_TIMEOUT)),
            Some(DEFAULT_HTTP_TIMEOUT)
        );
        assert_eq!(request_timeout(true, Some(DEFAULT_HTTP_TIMEOUT)), None);
    }

    #[test]
    fn embedding_capability_resolver_overrides_builtin_answer() {
        use anyllm::{EmbeddingCapability, EmbeddingProvider};

        let provider = Provider::new("test-key")
            .unwrap()
            .with_embedding_capabilities(|_model: &str, capability| {
                if capability == EmbeddingCapability::BatchInput {
                    Some(CapabilitySupport::Unsupported)
                } else {
                    None
                }
            });

        assert_eq!(
            provider.embedding_capability("text-embedding-004", EmbeddingCapability::BatchInput),
            CapabilitySupport::Unsupported,
        );
        // OutputDimensions falls through to builtin
        assert_eq!(
            provider.embedding_capability("text-embedding-004", EmbeddingCapability::OutputDimensions),
            CapabilitySupport::Supported,
        );
    }
}
