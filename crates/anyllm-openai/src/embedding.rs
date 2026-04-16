use anyllm::{
    CapabilitySupport, EmbeddingCapability, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse,
    Result,
};
use anyllm_openai_compat::{
    EmbeddingRequestOptions as CompatEmbeddingRequestOptions, EmbeddingsResponse,
    from_embeddings_response, send_embeddings_request, to_embeddings_request,
};

use crate::EmbeddingRequestOptions;
use crate::Provider;
use crate::error::{map_http_error, map_response_deserialize_error, map_transport_error};

impl Provider {
    async fn send_embeddings_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<reqwest::Response> {
        let provider_options = request.option::<EmbeddingRequestOptions>();
        let mut compat_options = CompatEmbeddingRequestOptions::default();
        compat_options.user = provider_options.and_then(|o| o.user.clone());
        let api_request = to_embeddings_request(request, &compat_options)?;

        let config = self.transport_config();
        let url = format!("{}/embeddings", config.base_url);

        let response = send_embeddings_request(
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
                if let Some(timeout) = crate::request_timeout(false, self.inner.request_timeout) {
                    req = req.timeout(timeout);
                }

                req.body(body).send()
            },
            map_transport_error,
        )
        .await?;

        let status = response.status();
        if !status.is_success() {
            let request_id = anyllm_openai_compat::extract_request_id(
                response.headers(),
                &config.request_id_header_name,
            );
            let retry_after = anyllm_openai_compat::extract_retry_after(
                response.headers(),
                &config.retry_after_header_name,
            );
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
        let response = self.send_embeddings_request(request).await?;
        let api_response: EmbeddingsResponse = response
            .json()
            .await
            .map_err(map_response_deserialize_error)?;
        from_embeddings_response(api_response, |_, _| {})
    }

    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        if let Some(support) = self
            .inner
            .embedding_capability_resolver
            .as_ref()
            .and_then(|resolver| resolver.embedding_capability(model, capability))
        {
            support
        } else {
            self.builtin_embedding_capability(model, capability)
        }
    }
}
