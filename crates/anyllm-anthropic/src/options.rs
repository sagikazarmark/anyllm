use anyllm::ExtraMap;

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct ChatRequestOptions {
    pub top_k: Option<u32>,
    pub metadata: Option<ExtraMap>,
    pub anthropic_beta: Vec<String>,
    /// Provider-specific thinking budget override (in tokens). When set, takes
    /// precedence over `ChatRequest.reasoning.budget_tokens` and `effort`.
    pub budget_tokens: Option<u32>,
}
