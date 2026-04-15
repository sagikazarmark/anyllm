//! `ChatProvider` implementation for Cloudflare Workers AI via the `worker::Ai` binding.

#[cfg(feature = "extract")]
use anyllm::ExtractExt;
use anyllm::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ProviderIdentity, Result,
};

use crate::Provider;
use crate::error::map_worker_error;
use crate::streaming::byte_stream_to_chat_stream;
use crate::wire;

impl ProviderIdentity for Provider {
    fn provider_name(&self) -> &'static str {
        "cloudflare-worker"
    }
}

impl ChatProvider for Provider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let cf_request = wire::ChatRequest::try_from(request)?;

        let response: wire::ChatResponse = self
            .ai
            .run(&request.model, &cf_request)
            .await
            .map_err(map_worker_error)?;

        response.try_into()
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        wire::reject_unsupported_streaming_request_features(request)?;

        let mut cf_request = wire::ChatRequest::try_from(request)?;
        cf_request.stream = Some(true);

        let byte_stream = self
            .ai
            .run_bytes(&request.model, &cf_request)
            .await
            .map_err(map_worker_error)?;

        Ok(byte_stream_to_chat_stream(byte_stream))
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        if let Some(support) = self
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

#[cfg(feature = "extract")]
impl ExtractExt for Provider {}
