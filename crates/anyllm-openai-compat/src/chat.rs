use anyllm::{
    CapabilitySupport, ChatCapability, ChatRequest, ChatResponse, ChatStream, EmbeddingCapability,
    EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, Result,
};

use crate::embedding::{
    EmbeddingRequestOptions, from_embeddings_response, send_embeddings_request,
    to_embeddings_request,
};
use crate::{
    Provider, RequestOptions, extract_request_id, extract_retry_after, from_api_response,
    map_http_error, map_response_deserialize_error, map_stream_error, map_transport_error,
    send_chat_completion_request, sse_to_stream, to_chat_completion_request,
};

impl Provider {
    async fn send_request(
        &self,
        request: &ChatRequest,
        stream: bool,
    ) -> anyllm::Result<reqwest::Response> {
        let compat_options = RequestOptions::default();
        let api_request = to_chat_completion_request(request, stream, &compat_options)?;
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

impl anyllm::ChatProvider for Provider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> anyllm::Result<ChatResponse> {
        let response = self.send_request(request, false).await?;

        let api_response: crate::ChatCompletionResponse = response
            .json()
            .await
            .map_err(map_response_deserialize_error)?;

        from_api_response(api_response, |_, _| {})
    }

    async fn chat_stream(&self, request: &ChatRequest) -> anyllm::Result<ChatStream> {
        let response = self.send_request(request, true).await?;
        Ok(sse_to_stream(response.bytes_stream(), map_stream_error))
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

impl Provider {
    async fn send_embeddings_http(
        &self,
        api_request: &crate::embedding::EmbeddingsRequest,
    ) -> Result<reqwest::Response> {
        let config = &self.inner.transport;
        let url = config.embeddings_url();

        let response = send_embeddings_request(
            api_request,
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

impl EmbeddingProvider for Provider {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let options = request
            .option::<EmbeddingRequestOptions>()
            .cloned()
            .unwrap_or_default();
        let api_request = to_embeddings_request(request, &options)?;
        let response = self.send_embeddings_http(&api_request).await?;
        let api_response: crate::embedding::EmbeddingsResponse = response
            .json()
            .await
            .map_err(map_response_deserialize_error)?;
        from_embeddings_response(api_response, |_, _| {})
    }

    fn embedding_capability(
        &self,
        _model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        self.inner
            .embedding_capabilities
            .get(&capability)
            .copied()
            .unwrap_or(CapabilitySupport::Unknown)
    }
}

impl anyllm::ProviderIdentity for Provider {
    fn provider_name(&self) -> &'static str {
        self.inner.provider_name
    }
}

#[cfg(feature = "extract")]
impl anyllm::ExtractExt for Provider {}
