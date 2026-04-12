use crate::OpenAIReasoningEffort;
use anyllm::{ExtraMap, ResponseMetadataType};

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct ChatRequestOptions {
    /// Stable end-user identifier forwarded through OpenAI's `user` field.
    pub user: Option<String>,
    /// OpenAI-specific service tier hint such as `flex`.
    pub service_tier: Option<String>,
    /// Whether OpenAI may store the request/response for product improvements.
    pub store: Option<bool>,
    /// OpenAI-specific metadata forwarded as request JSON.
    pub metadata: Option<ExtraMap>,
    /// Provider-specific reasoning effort override. When set, takes precedence
    /// over `ChatRequest.reasoning`.
    pub reasoning_effort: Option<OpenAIReasoningEffort>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ChatResponseMetadata {
    /// OpenAI system fingerprint returned with the response, when present.
    pub system_fingerprint: Option<String>,
}

impl ResponseMetadataType for ChatResponseMetadata {
    const KEY: &'static str = "openai";
}
