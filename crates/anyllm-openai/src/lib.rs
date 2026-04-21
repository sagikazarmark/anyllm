use anyllm::{CapabilitySupport, ChatCapability, ChatCapabilityResolver, Error, Result};
use std::{sync::Arc, time::Duration};

mod chat;
#[cfg(test)]
mod conformance_tests;
mod embedding;
mod error;
mod options;
mod streaming;
mod wire;

pub use anyllm_openai_compat::OpenAIReasoningEffort;
use anyllm_openai_compat::TransportConfig;
pub use options::{ChatRequestOptions, ChatResponseMetadata, EmbeddingRequestOptions};

#[cfg(feature = "http-tracing")]
type HttpClient = reqwest_middleware::ClientWithMiddleware;
#[cfg(not(feature = "http-tracing"))]
type HttpClient = reqwest::Client;

/// OpenAI API provider implementing `anyllm::ChatProvider`.
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
    pub(crate) organization: Option<String>,
    pub(crate) project: Option<String>,
    pub(crate) chat_capability_resolver: Option<Arc<dyn ChatCapabilityResolver>>,
}

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(120);

fn supports_openai_structured_output(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    model.starts_with("gpt-4.1")
        || model.starts_with("gpt-4o")
        || model.starts_with("gpt-5")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
}

fn supports_openai_vision(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    model.starts_with("gpt-4.1")
        || model.starts_with("gpt-4o")
        || model.starts_with("gpt-5")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
}

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

fn optional_nonempty(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.trim().is_empty())
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
    fn transport_config(&self) -> TransportConfig {
        TransportConfig {
            base_url: self.inner.base_url.clone(),
            chat_completions_path: "/chat/completions".into(),
            embeddings_path: "/embeddings".into(),
            auth_header_name: "authorization".into(),
            auth_header_value: format!("Bearer {}", self.inner.api_key),
            organization_header: self
                .inner
                .organization
                .clone()
                .map(|value| ("openai-organization".into(), value)),
            project_header: self
                .inner
                .project
                .clone()
                .map(|value| ("openai-project".into(), value)),
            request_id_header_name: "x-request-id".into(),
            retry_after_header_name: "retry-after".into(),
        }
    }

    fn default_http_client() -> HttpClient {
        let base = reqwest::Client::builder()
            .build()
            .expect("default OpenAI reqwest client config should be valid");
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
        model: &str,
        capability: ChatCapability,
    ) -> CapabilitySupport {
        match capability {
            ChatCapability::ToolCalls
            | ChatCapability::ParallelToolCalls
            | ChatCapability::Streaming
            | ChatCapability::NativeStreaming => CapabilitySupport::Supported,
            ChatCapability::ReasoningConfig => CapabilitySupport::Unknown,
            ChatCapability::ImageInput | ChatCapability::ImageDetail => {
                if supports_openai_vision(model) {
                    CapabilitySupport::Supported
                } else {
                    CapabilitySupport::Unknown
                }
            }
            ChatCapability::StructuredOutput => {
                if supports_openai_structured_output(model) {
                    CapabilitySupport::Supported
                } else {
                    CapabilitySupport::Unknown
                }
            }
            ChatCapability::ImageOutput
            | ChatCapability::ImageReplay
            | ChatCapability::ReasoningOutput
            | ChatCapability::ReasoningReplay => CapabilitySupport::Unsupported,
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
                organization: None,
                project: None,
                chat_capability_resolver: None,
            }),
        })
    }

    /// Create from environment variables.
    ///
    /// Required: `OPENAI_API_KEY`
    /// Optional: `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`
    pub fn from_env() -> Result<Self> {
        let api_key = required_env_var("OPENAI_API_KEY")?;
        let base_url = std::env::var("OPENAI_BASE_URL").ok();
        let organization = std::env::var("OPENAI_ORG_ID").ok();
        let project = std::env::var("OPENAI_PROJECT_ID").ok();

        let base_url = match base_url {
            Some(url) if !url.trim().is_empty() => normalize_base_url(url),
            _ => DEFAULT_BASE_URL.to_string(),
        };

        Ok(Self {
            inner: Arc::new(Inner {
                client: Self::default_http_client(),
                request_timeout: Some(DEFAULT_HTTP_TIMEOUT),
                api_key,
                base_url,
                organization: optional_nonempty(organization),
                project: optional_nonempty(project),
                chat_capability_resolver: None,
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
                organization: self.inner.organization.clone(),
                project: self.inner.project.clone(),
                chat_capability_resolver: Some(Arc::new(resolver)),
            }),
        }
    }

    /// Builder for full configuration.
    #[must_use]
    pub fn builder() -> ProviderBuilder {
        ProviderBuilder {
            api_key: None,
            base_url: None,
            organization: None,
            project: None,
            client: None,
            request_timeout: Some(DEFAULT_HTTP_TIMEOUT),
        }
    }
}

/// Builder for configuring an OpenAI provider.
pub struct ProviderBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    organization: Option<String>,
    project: Option<String>,
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
    pub fn organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }
    #[must_use]
    pub fn project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
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
                organization: optional_nonempty(self.organization),
                project: optional_nonempty(self.project),
                chat_capability_resolver: None,
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
        let _api_key = EnvVarGuard::set("OPENAI_API_KEY", Some("   "));
        let _base_url = EnvVarGuard::set("OPENAI_BASE_URL", None);
        let _organization = EnvVarGuard::set("OPENAI_ORG_ID", None);
        let _project = EnvVarGuard::set("OPENAI_PROJECT_ID", None);

        let err = match Provider::from_env() {
            Ok(_) => panic!("expected empty API key to be rejected"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::Auth(message) if message == "OPENAI_API_KEY not set"));
    }

    #[test]
    fn chat_capability_resolver_overrides_builtin_answer() {
        let provider =
            Provider::new("test-key")
                .unwrap()
                .with_chat_capabilities(|model: &str, capability| {
                    if model == "gpt-4.1-mini" && capability == ChatCapability::StructuredOutput {
                        Some(CapabilitySupport::Unknown)
                    } else {
                        None
                    }
                });

        assert_eq!(
            provider.chat_capability("gpt-4.1-mini", ChatCapability::StructuredOutput),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            provider.chat_capability("gpt-4.1", ChatCapability::StructuredOutput),
            CapabilitySupport::Supported
        );
    }

    #[test]
    fn builtin_openai_capabilities_are_conservative_for_model_specific_features() {
        let provider = Provider::new("test-key").unwrap();

        assert_eq!(
            provider.chat_capability("gpt-4o", ChatCapability::ToolCalls),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.chat_capability("gpt-4o", ChatCapability::ParallelToolCalls),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.chat_capability("gpt-4o", ChatCapability::Streaming),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.chat_capability("gpt-4o", ChatCapability::NativeStreaming),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.chat_capability("gpt-4o", ChatCapability::StructuredOutput),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.chat_capability("gpt-4o", ChatCapability::ImageInput),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.chat_capability("gpt-4o", ChatCapability::ImageDetail),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.chat_capability("gpt-4.1", ChatCapability::ImageInput),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.chat_capability("gpt-3.5-turbo", ChatCapability::ImageInput),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            provider.chat_capability("gpt-3.5-turbo", ChatCapability::ImageDetail),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            provider.chat_capability("gpt-4o", ChatCapability::ReasoningConfig),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            provider.chat_capability("gpt-3.5-turbo", ChatCapability::StructuredOutput),
            CapabilitySupport::Unknown
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
            provider.chat_capability("gpt-4.1", ChatCapability::ToolCalls),
            CapabilitySupport::Supported
        );
    }

    #[test]
    fn base_url_is_normalized_when_constructed() {
        let provider =
            Provider::new_with_base_url("test-key", "https://example.com/v1///").unwrap();
        assert_eq!(provider.inner.base_url, "https://example.com/v1");
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
            .base_url("https://example.com/v1///")
            .build()
            .unwrap();

        assert_eq!(provider.inner.base_url, "https://example.com/v1");
    }

    #[test]
    fn builder_discards_blank_optional_headers() {
        let provider = Provider::builder()
            .api_key("test-key")
            .organization("   ")
            .project("")
            .build()
            .unwrap();

        assert_eq!(provider.inner.organization, None);
        assert_eq!(provider.inner.project, None);
    }

    #[test]
    fn default_timeout_is_only_applied_to_non_streaming_requests() {
        assert_eq!(
            request_timeout(false, Some(DEFAULT_HTTP_TIMEOUT)),
            Some(DEFAULT_HTTP_TIMEOUT)
        );
        assert_eq!(request_timeout(true, Some(DEFAULT_HTTP_TIMEOUT)), None);
    }
}
