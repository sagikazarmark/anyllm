#[cfg(feature = "extract")]
use anyllm::ExtractExt;
use anyllm::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ProviderIdentity, Result,
};

use crate::ChatRequestOptions;
use crate::Provider;
use crate::error::{map_http_error, map_response_deserialize_error, map_transport_error};
use crate::streaming::sse_to_stream;
use crate::wire;

fn parse_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<std::time::Duration> {
    let seconds = headers
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<f64>().ok())?;

    if !seconds.is_finite() || seconds.is_sign_negative() {
        return None;
    }

    Some(std::time::Duration::from_secs_f64(seconds))
}

impl Provider {
    /// Send an HTTP request to the Anthropic Messages API.
    ///
    /// Handles serialization, headers, and error mapping.
    async fn send_request(&self, request: &ChatRequest, stream: bool) -> Result<reqwest::Response> {
        let mut api_request = wire::CreateMessageRequest::try_from(request)?;
        api_request.stream = stream;

        let url = format!("{}/v1/messages", self.inner.base_url);

        let body = serde_json::to_string(&api_request).map_err(anyllm::Error::from)?;

        let mut req = self
            .inner
            .client
            .post(&url)
            .header("x-api-key", &self.inner.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        if let Some(timeout) = crate::request_timeout(stream, self.inner.request_timeout) {
            req = req.timeout(timeout);
        }

        if let Some(opts) = request.option::<ChatRequestOptions>() {
            for beta in &opts.anthropic_beta {
                req = req.header("anthropic-beta", beta);
            }
        }

        let response = req.body(body).send().await.map_err(map_transport_error)?;

        let status = response.status();

        if !status.is_success() {
            let request_id = response
                .headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok())
                .map(String::from);

            let retry_after = parse_retry_after_header(response.headers());

            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error body".to_string());

            return Err(map_http_error(
                status.as_u16(),
                &error_body,
                request_id,
                retry_after,
            ));
        }

        Ok(response)
    }
}

impl ChatProvider for Provider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let response = self.send_request(request, false).await?;

        let api_response: wire::CreateMessageResponse = response
            .json()
            .await
            .map_err(map_response_deserialize_error)?;

        api_response.try_into()
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let response = self.send_request(request, true).await?;
        Ok(sse_to_stream(response.bytes_stream()))
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        if let Some(support) = self
            .inner
            .chat_capability_resolver
            .as_ref()
            .and_then(|resolver| resolver.chat_capability(model, capability))
        {
            support
        } else {
            self.builtin_chat_capability(model, capability)
        }
    }
}

impl ProviderIdentity for Provider {
    fn provider_name(&self) -> &'static str {
        "anthropic"
    }
}

#[cfg(feature = "extract")]
impl ExtractExt for Provider {}
