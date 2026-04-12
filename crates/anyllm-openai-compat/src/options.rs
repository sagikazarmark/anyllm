use anyllm::{ExtraMap, ReasoningEffort};

/// OpenAI-compatible reasoning effort levels.
///
/// This extends the core `ReasoningEffort` enum with provider-specific values
/// like `XHigh` that only exist in the OpenAI API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIReasoningEffort {
    Low,
    Medium,
    High,
    XHigh,
}

impl std::fmt::Display for OpenAIReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::XHigh => write!(f, "xhigh"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RequestOptions {
    pub seed: Option<u64>,
    pub parallel_tool_calls: Option<bool>,
    pub user: Option<String>,
    pub service_tier: Option<String>,
    pub store: Option<bool>,
    pub metadata: Option<ExtraMap>,
    pub extra: ExtraMap,
    /// Provider-specific reasoning effort override. When set, takes precedence
    /// over the `ChatRequest.reasoning` derivation.
    pub reasoning_effort: Option<OpenAIReasoningEffort>,
}

pub(crate) fn reasoning_effort_from_config(
    config: &anyllm::ReasoningConfig,
) -> Option<&'static str> {
    if !config.enabled {
        return None;
    }

    if let Some(effort) = &config.effort {
        if matches!(effort, ReasoningEffort::Low) {
            return Some("low");
        }
        if matches!(effort, ReasoningEffort::Medium) {
            return Some("medium");
        }
        return Some("high");
    }

    config.budget_tokens.map(|budget| {
        if budget < 1024 {
            "low"
        } else if budget < 4096 {
            "medium"
        } else {
            "high"
        }
    })
}
