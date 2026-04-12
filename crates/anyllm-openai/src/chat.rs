#[cfg(feature = "extract")]
use anyllm::ExtractExt;
use anyllm::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream, Result,
};
use anyllm_openai_compat::{extract_request_id, extract_retry_after, send_chat_completion_request};

use crate::Provider;
use crate::error::{map_http_error, map_response_deserialize_error, map_transport_error};
use crate::streaming::sse_to_stream;
use crate::wire::{ChatCompletionResponse, from_api_response, to_api_request};

impl Provider {
    /// Send an HTTP request to the OpenAI Chat Completions API.
    ///
    /// Handles serialization, headers, and error mapping.
    async fn send_request(&self, request: &ChatRequest, stream: bool) -> Result<reqwest::Response> {
        let api_request = to_api_request(request, stream)?;
        let config = self.transport_config();
        let url = config.chat_completions_url();

        let response = send_chat_completion_request(
            &api_request,
            |body| {
                let mut req = self
                    .inner
                    .client
                    .post(&url)
                    .header(&config.auth_header_name, &config.auth_header_value)
                    .header("content-type", "application/json");

                if let Some((ref name, ref value)) = config.organization_header {
                    req = req.header(name, value);
                }

                if let Some((ref name, ref value)) = config.project_header {
                    req = req.header(name, value);
                }

                if let Some(timeout) = crate::request_timeout(stream, self.inner.request_timeout) {
                    req = req.timeout(timeout);
                }

                req.body(body).send()
            },
            map_transport_error,
        )
        .await?;

        let status = response.status();

        if !status.is_success() {
            let request_id = extract_request_id(response.headers(), &config.request_id_header_name);

            let retry_after =
                extract_retry_after(response.headers(), &config.retry_after_header_name);

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

        let api_response: ChatCompletionResponse = response
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
        "openai"
    }
}

#[cfg(feature = "extract")]
impl ExtractExt for Provider {}
