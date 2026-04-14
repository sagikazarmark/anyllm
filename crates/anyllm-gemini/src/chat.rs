#[cfg(feature = "extract")]
use anyllm::ExtractExt;
use anyllm::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream, Result,
};

use crate::Provider;
use crate::error::{map_http_error, map_response_deserialize_error, map_transport_error};
use crate::streaming::sse_to_stream;
use crate::wire::{GenerateContentResponse, from_api_response, to_api_request};

fn parse_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<std::time::Duration> {
    let seconds = headers
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<f64>().ok())?;

    if !seconds.is_finite() || seconds.is_sign_negative() {
        return None;
    }

    Some(std::time::Duration::from_secs_f64(seconds))
}

impl Provider {
    /// Send an HTTP request to the Gemini generateContent API.
    async fn send_request(&self, request: &ChatRequest, stream: bool) -> Result<reqwest::Response> {
        let api_request = to_api_request(request)?;

        let url = if stream {
            format!(
                "{}/models/{}:streamGenerateContent?alt=sse",
                self.inner.base_url, request.model
            )
        } else {
            format!(
                "{}/models/{}:generateContent",
                self.inner.base_url, request.model
            )
        };

        let body = serde_json::to_string(&api_request).map_err(anyllm::Error::from)?;

        let mut request = self
            .inner
            .client
            .post(&url)
            .header("x-goog-api-key", &self.inner.api_key)
            .header("content-type", "application/json");

        if let Some(timeout) = crate::request_timeout(stream, self.inner.request_timeout) {
            request = request.timeout(timeout);
        }

        let response = request
            .body(body)
            .send()
            .await
            .map_err(map_transport_error)?;

        let status = response.status();

        if !status.is_success() {
            let retry_after = parse_retry_after_header(response.headers());

            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error body".to_string());

            return Err(map_http_error(status.as_u16(), &error_body, retry_after));
        }

        Ok(response)
    }
}

impl ChatProvider for Provider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let response = self.send_request(request, false).await?;

        let api_response: GenerateContentResponse = response
            .json()
            .await
            .map_err(map_response_deserialize_error)?;

        from_api_response(api_response)
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

    fn provider_name(&self) -> &'static str {
        "gemini"
    }
}

#[cfg(feature = "extract")]
impl ExtractExt for Provider {}
