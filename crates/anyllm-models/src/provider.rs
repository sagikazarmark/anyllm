use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::Model;

/// A provider entry from the `models.dev` registry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provider {
    /// Provider identifier (e.g. `"openai"`, `"anthropic"`).
    pub id: String,

    /// Human-readable provider name.
    pub name: String,

    /// Environment variable names for API keys.
    pub env: Vec<String>,

    /// NPM package name for the provider's SDK.
    pub npm: String,

    /// Base API endpoint URL.
    pub api: String,

    /// Documentation URL.
    pub doc: String,

    /// Models offered by this provider, keyed by model id.
    pub models: HashMap<String, Model>,
}
