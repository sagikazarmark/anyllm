use std::collections::HashMap;
use std::sync::Arc;

use anyllm::{CapabilitySupport, ChatCapability, ChatCapabilityResolver, Error, Result};

mod chat;
mod embedding;
mod error;
mod options;
pub mod providers;
mod streaming;
mod wire;

pub use embedding::{
    EmbeddingData, EmbeddingRequestOptions, EmbeddingsRequest, EmbeddingsResponse, EmbeddingsUsage,
    from_embeddings_response, send_embeddings_request, to_embeddings_request,
};
pub use error::{
    map_http_error, map_response_deserialize_error, map_stream_error, map_transport_error,
};
pub use options::{OpenAIReasoningEffort, RequestOptions};
pub use streaming::{SseState, process_sse_data, sse_to_stream};
pub use wire::{
    ChatCompletionRequest, ChatCompletionResponse, from_api_response, parse_finish_reason,
    to_chat_completion_request,
};

/// A generic OpenAI-compatible chat provider.
///
/// Works with any API that speaks the OpenAI chat completions wire format
/// (Cloudflare Workers AI, Groq, Together, Fireworks, DeepInfra, etc.).
///
/// Construct via pre-configured provider factories in [`providers`] or use
/// [`Provider::builder()`] for custom endpoints.
///
/// Clone is cheap — internals are wrapped in `Arc`.
#[derive(Clone)]
pub struct Provider {
    pub(crate) inner: Arc<Inner>,
}

pub(crate) struct Inner {
    pub(crate) client: reqwest::Client,
    pub(crate) transport: TransportConfig,
    pub(crate) chat_capabilities: HashMap<ChatCapability, CapabilitySupport>,
    pub(crate) chat_capability_resolver: Option<Arc<dyn ChatCapabilityResolver>>,
    pub(crate) provider_name: &'static str,
}

fn normalize_base_url(url: impl Into<String>) -> String {
    url.into().trim().trim_end_matches('/').to_string()
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
        None => Err(Error::InvalidRequest("base_url is required".into())),
    }
}

impl Provider {
    fn default_http_client() -> reqwest::Client {
        reqwest::Client::builder()
            .build()
            .expect("default OpenAI-compatible reqwest client config should be valid")
    }

    /// Create a builder for full configuration.
    pub fn builder() -> ProviderBuilder {
        ProviderBuilder {
            base_url: None,
            chat_completions_path: None,
            auth_header_name: None,
            auth_header_value: None,
            organization_header: None,
            project_header: None,
            request_id_header_name: None,
            retry_after_header_name: None,
            chat_capabilities: HashMap::new(),
            provider_name: None,
            client: None,
        }
    }

    pub(crate) fn transport_config(&self) -> &TransportConfig {
        &self.inner.transport
    }

    /// Install a resolver consulted before the provider's configured capability logic.
    #[must_use]
    pub fn with_chat_capabilities(self, resolver: impl ChatCapabilityResolver) -> Self {
        Self {
            inner: Arc::new(Inner {
                client: self.inner.client.clone(),
                transport: self.inner.transport.clone(),
                chat_capabilities: self.inner.chat_capabilities.clone(),
                chat_capability_resolver: Some(Arc::new(resolver)),
                provider_name: self.inner.provider_name,
            }),
        }
    }

    pub(crate) fn builtin_chat_capability(
        &self,
        _model: &str,
        capability: ChatCapability,
    ) -> CapabilitySupport {
        self.inner
            .chat_capabilities
            .get(&capability)
            .copied()
            .unwrap_or(CapabilitySupport::Unknown)
    }
}

/// Builder for configuring a [`Provider`].
pub struct ProviderBuilder {
    base_url: Option<String>,
    chat_completions_path: Option<String>,
    auth_header_name: Option<String>,
    auth_header_value: Option<String>,
    organization_header: Option<(String, String)>,
    project_header: Option<(String, String)>,
    request_id_header_name: Option<String>,
    retry_after_header_name: Option<String>,
    chat_capabilities: HashMap<ChatCapability, CapabilitySupport>,
    provider_name: Option<&'static str>,
    client: Option<reqwest::Client>,
}

impl ProviderBuilder {
    /// The base URL for the API (e.g. `https://api.groq.com/openai/v1`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Path appended to base_url for chat completions.
    /// Defaults to `/chat/completions`.
    pub fn chat_completions_path(mut self, path: impl Into<String>) -> Self {
        self.chat_completions_path = Some(path.into());
        self
    }

    /// Set bearer token auth (`Authorization: Bearer {token}`).
    /// This is the most common auth style for OpenAI-compatible APIs.
    pub fn bearer_token(mut self, token: impl Into<String>) -> Self {
        self.auth_header_name = Some("authorization".into());
        self.auth_header_value = Some(format!("Bearer {}", token.into()));
        self
    }

    /// Set custom auth header name and value.
    pub fn auth_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.auth_header_name = Some(name.into());
        self.auth_header_value = Some(value.into());
        self
    }

    /// Set an additional organization header.
    pub fn organization_header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.organization_header = Some((name.into(), value.into()));
        self
    }

    /// Set an additional project header.
    pub fn project_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.project_header = Some((name.into(), value.into()));
        self
    }

    /// Header name for extracting request IDs from responses.
    /// Defaults to `x-request-id`.
    pub fn request_id_header_name(mut self, name: impl Into<String>) -> Self {
        self.request_id_header_name = Some(name.into());
        self
    }

    /// Header name for extracting retry-after from responses.
    /// Defaults to `retry-after`.
    pub fn retry_after_header_name(mut self, name: impl Into<String>) -> Self {
        self.retry_after_header_name = Some(name.into());
        self
    }

    /// Set support information for a chat capability.
    pub fn chat_capability(
        mut self,
        capability: ChatCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.chat_capabilities.insert(capability, support);
        self
    }

    /// Set multiple chat capability support values for this provider.
    pub fn chat_capabilities<I>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = (ChatCapability, CapabilitySupport)>,
    {
        self.chat_capabilities.extend(capabilities);
        self
    }

    /// Set the provider name (used in logging and error messages).
    pub fn provider_name(mut self, name: &'static str) -> Self {
        self.provider_name = Some(name);
        self
    }

    /// Set a custom reqwest client.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Build the provider.
    pub fn build(self) -> Result<Provider> {
        let auth_header_value = self.auth_header_value.ok_or_else(|| {
            Error::InvalidRequest("auth is required — use bearer_token() or auth_header()".into())
        })?;

        let transport = TransportConfig {
            base_url: builder_base_url(self.base_url)?,
            chat_completions_path: self
                .chat_completions_path
                .unwrap_or_else(|| "/chat/completions".into()),
            auth_header_name: self
                .auth_header_name
                .unwrap_or_else(|| "authorization".into()),
            auth_header_value: required_builder_value(
                "auth_header_value",
                Some(auth_header_value),
            )?,
            organization_header: self.organization_header,
            project_header: self.project_header,
            request_id_header_name: self
                .request_id_header_name
                .unwrap_or_else(|| "x-request-id".into()),
            retry_after_header_name: self
                .retry_after_header_name
                .unwrap_or_else(|| "retry-after".into()),
        };

        Ok(Provider {
            inner: Arc::new(Inner {
                client: self.client.unwrap_or_else(Provider::default_http_client),
                transport,
                chat_capabilities: self.chat_capabilities,
                chat_capability_resolver: None,
                provider_name: self.provider_name.unwrap_or("openai_compat"),
            }),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportConfig {
    pub base_url: String,
    pub chat_completions_path: String,
    pub auth_header_name: String,
    pub auth_header_value: String,
    pub organization_header: Option<(String, String)>,
    pub project_header: Option<(String, String)>,
    pub request_id_header_name: String,
    pub retry_after_header_name: String,
}

impl TransportConfig {
    pub fn chat_completions_url(&self) -> String {
        format!("{}{}", self.base_url, self.chat_completions_path)
    }
}

pub fn extract_request_id(
    headers: &reqwest::header::HeaderMap,
    header_name: &str,
) -> Option<String> {
    headers
        .get(header_name)
        .and_then(|value| value.to_str().ok())
        .map(String::from)
}

pub fn extract_retry_after(
    headers: &reqwest::header::HeaderMap,
    header_name: &str,
) -> Option<std::time::Duration> {
    headers
        .get(header_name)
        .and_then(|value| value.to_str().ok())
        .and_then(parse_retry_after_value)
}

fn parse_retry_after_value(value: &str) -> Option<std::time::Duration> {
    let seconds = value.parse::<f64>().ok()?;
    if !seconds.is_finite() || seconds.is_sign_negative() {
        return None;
    }

    Some(std::time::Duration::from_secs_f64(seconds))
}

pub async fn send_chat_completion_request<E, Fut, F, M>(
    api_request: &ChatCompletionRequest,
    send: F,
    map_transport_error: M,
) -> Result<reqwest::Response>
where
    F: FnOnce(String) -> Fut,
    Fut: std::future::Future<Output = std::result::Result<reqwest::Response, E>>,
    M: Fn(E) -> Error,
{
    let body = serde_json::to_string(api_request).map_err(Error::from)?;

    send(body).await.map_err(map_transport_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyllm::ChatProvider;
    use serde_json::json;

    #[test]
    fn transport_config_builds_chat_completions_url() {
        let config = TransportConfig {
            base_url: "https://example.com/v1".into(),
            chat_completions_path: "/chat/completions".into(),
            auth_header_name: "authorization".into(),
            auth_header_value: "Bearer sk-test".into(),
            organization_header: None,
            project_header: None,
            request_id_header_name: "x-request-id".into(),
            retry_after_header_name: "retry-after".into(),
        };

        assert_eq!(
            config.chat_completions_url(),
            "https://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn extracts_request_id_from_configured_header() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-custom-request-id", "req_123".parse().unwrap());

        assert_eq!(
            extract_request_id(&headers, "x-custom-request-id").as_deref(),
            Some("req_123")
        );
    }

    #[test]
    fn extracts_retry_after_from_configured_header() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-retry-after", "2.5".parse().unwrap());

        assert_eq!(
            extract_retry_after(&headers, "x-retry-after"),
            Some(std::time::Duration::from_secs_f64(2.5))
        );
    }

    #[test]
    fn ignores_negative_retry_after_from_configured_header() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-retry-after", "-1".parse().unwrap());

        assert_eq!(extract_retry_after(&headers, "x-retry-after"), None);
    }

    #[test]
    fn ignores_non_finite_retry_after_from_configured_header() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-retry-after", "NaN".parse().unwrap());

        assert_eq!(extract_retry_after(&headers, "x-retry-after"), None);
    }

    #[test]
    fn response_conversion_supports_metadata_hook() {
        let response: ChatCompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello"
                },
                "finish_reason": "stop"
            }],
            "model": "gpt-4o",
            "system_fingerprint": "fp_test"
        }))
        .unwrap();

        #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
        struct DemoMetadata {
            fingerprint: String,
        }

        impl anyllm::ResponseMetadataType for DemoMetadata {
            const KEY: &'static str = "demo";
        }

        let converted = from_api_response(response, |response, metadata| {
            if let Some(fp) = &response.system_fingerprint {
                metadata.insert(DemoMetadata {
                    fingerprint: fp.clone(),
                });
            }
        })
        .unwrap();

        assert_eq!(converted.text().as_deref(), Some("hello"));
        assert_eq!(
            serde_json::to_value(&converted.metadata).unwrap(),
            json!({
                "demo": {"fingerprint": "fp_test"}
            })
        );
    }

    #[test]
    fn builder_requires_base_url() {
        let result = Provider::builder().bearer_token("token").build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_empty_base_url() {
        let result = Provider::builder()
            .base_url("   ")
            .bearer_token("token")
            .build();
        assert!(
            matches!(result, Err(Error::InvalidRequest(message)) if message == "base_url cannot be empty")
        );
    }

    #[test]
    fn builder_normalizes_base_url() {
        let provider = Provider::builder()
            .base_url(" https://api.example.com/v1/ ")
            .bearer_token("token")
            .build()
            .unwrap();
        assert_eq!(
            provider.transport_config().base_url,
            "https://api.example.com/v1"
        );
    }

    #[test]
    fn builder_requires_auth() {
        let result = Provider::builder()
            .base_url("https://example.com/v1")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn custom_auth_header() {
        let provider = Provider::builder()
            .base_url("https://api.example.com/v1")
            .auth_header("x-api-key", "my-secret")
            .build()
            .unwrap();

        let config = provider.transport_config();
        assert_eq!(config.auth_header_name, "x-api-key");
        assert_eq!(config.auth_header_value, "my-secret");
    }

    #[test]
    fn chat_capability_resolver_takes_precedence_over_configured_capabilities() {
        let provider = Provider::builder()
            .base_url("https://api.example.com/v1")
            .bearer_token("token")
            .chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Supported,
            )
            .build()
            .unwrap()
            .with_chat_capabilities(|model: &str, capability| {
                if model == "legacy" && capability == ChatCapability::StructuredOutput {
                    Some(CapabilitySupport::Unknown)
                } else {
                    None
                }
            });

        assert_eq!(
            provider.chat_capability("legacy", ChatCapability::StructuredOutput),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            provider.chat_capability("modern", ChatCapability::StructuredOutput),
            CapabilitySupport::Supported
        );
    }
}
