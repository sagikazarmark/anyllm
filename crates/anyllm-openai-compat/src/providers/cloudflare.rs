use anyllm::{CapabilitySupport, ChatCapability, EmbeddingCapability};

use crate::{Provider, ProviderBuilder};

fn required_value<'a>(name: &'static str, value: &'a str) -> anyllm::Result<&'a str> {
    let value = value.trim();
    if value.is_empty() {
        return Err(anyllm::Error::InvalidRequest(format!("{name} is required")));
    }

    Ok(value)
}

fn required_env_var(name: &'static str) -> anyllm::Result<String> {
    let value = std::env::var(name).map_err(|_| anyllm::Error::Auth(format!("{name} not set")))?;
    let value = value.trim();
    if value.is_empty() {
        return Err(anyllm::Error::Auth(format!("{name} not set")));
    }

    Ok(value.to_string())
}

/// Factory for Cloudflare Workers AI via the OpenAI-compatible REST API.
///
/// Uses the endpoint:
/// `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions`
///
/// # Environment Variables
///
/// - `CLOUDFLARE_ACCOUNT_ID` — your Cloudflare account ID (required for `from_env`)
/// - `CLOUDFLARE_API_TOKEN` — a Cloudflare API token with Workers AI permissions (required for `from_env`)
///
/// # Example
///
/// ```rust,no_run
/// use anyllm::prelude::*;
/// use anyllm_openai_compat::providers::Cloudflare;
///
/// # async fn example() -> anyllm::Result<()> {
/// let provider = Cloudflare::new("your-account-id", "your-api-token")?;
/// let response = provider.chat(
///     &ChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
///         .message(Message::user("What is Rust?"))
/// ).await?;
/// println!("{}", response.text_or_empty());
/// # Ok(())
/// # }
/// ```
pub struct Cloudflare;

impl Cloudflare {
    /// Create a Cloudflare Workers AI provider with the given credentials.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(account_id: &str, api_token: &str) -> anyllm::Result<Provider> {
        Self::builder(account_id, api_token)?.build()
    }

    /// Create a builder pre-configured for Cloudflare, allowing customization.
    ///
    /// Use this when you need to override defaults (e.g., custom client, AI Gateway URL).
    pub fn builder(account_id: &str, api_token: &str) -> anyllm::Result<ProviderBuilder> {
        let account_id = required_value("account_id", account_id)?;
        let api_token = required_value("api_token", api_token)?;

        Ok(Provider::builder()
            .base_url(format!(
                "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
            ))
            .bearer_token(api_token)
            .provider_name("cloudflare")
            .chat_capabilities(Self::default_chat_capabilities())
            .embedding_capabilities(Self::default_embedding_capabilities()))
    }

    /// Create a Cloudflare Workers AI provider from environment variables.
    ///
    /// Reads `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`.
    pub fn from_env() -> anyllm::Result<Provider> {
        let account_id = required_env_var("CLOUDFLARE_ACCOUNT_ID")?;
        let api_token = required_env_var("CLOUDFLARE_API_TOKEN")?;

        Self::new(&account_id, &api_token)
    }

    fn default_chat_capabilities() -> [(ChatCapability, CapabilitySupport); 4] {
        [
            (ChatCapability::ToolCalls, CapabilitySupport::Supported),
            (ChatCapability::Streaming, CapabilitySupport::Supported),
            (ChatCapability::NativeStreaming, CapabilitySupport::Supported),
            (
                ChatCapability::StructuredOutput,
                CapabilitySupport::Supported,
            ),
        ]
    }

    fn default_embedding_capabilities() -> [(EmbeddingCapability, CapabilitySupport); 1] {
        [(
            EmbeddingCapability::BatchInput,
            CapabilitySupport::Supported,
        )]
    }
}

#[cfg(test)]
mod tests {
    use super::Cloudflare;
    use anyllm::ProviderIdentity;

    #[test]
    fn new_rejects_blank_account_id() {
        let result = Cloudflare::new("   ", "token");
        assert!(matches!(
            result,
            Err(anyllm::Error::InvalidRequest(message)) if message == "account_id is required"
        ));
    }

    #[test]
    fn new_rejects_blank_api_token() {
        let result = Cloudflare::new("account", "   ");
        assert!(matches!(
            result,
            Err(anyllm::Error::InvalidRequest(message)) if message == "api_token is required"
        ));
    }

    #[test]
    fn builder_accepts_valid_credentials() {
        let provider = Cloudflare::builder("account", "token")
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(provider.provider_name(), "cloudflare");
    }

    #[test]
    fn cloudflare_defaults_include_embedding_capabilities() {
        use anyllm::{EmbeddingCapability, EmbeddingProvider};
        let provider = Cloudflare::new("account", "token").unwrap();
        assert_eq!(
            provider.embedding_capability(
                "@cf/baai/bge-base-en-v1.5",
                EmbeddingCapability::BatchInput,
            ),
            anyllm::CapabilitySupport::Supported,
        );
    }

    #[test]
    fn cloudflare_defaults_include_native_streaming() {
        use anyllm::{ChatCapability, ChatProvider};
        let provider = Cloudflare::new("account", "token").unwrap();
        assert_eq!(
            provider.chat_capability("@cf/meta/llama-3.1-8b-instruct", ChatCapability::NativeStreaming),
            anyllm::CapabilitySupport::Supported,
        );
    }
}
