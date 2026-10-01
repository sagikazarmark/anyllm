use serde::{Deserialize, Serialize};

use crate::ExtraMap;

fn extensions_are_absent_or_empty(extensions: &Option<ExtraMap>) -> bool {
    extensions.as_ref().is_none_or(ExtraMap::is_empty)
}

/// A tool (function) definition that the model may call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Tool {
    /// Tool name exposed to the model.
    pub name: String,
    /// Human-readable description used by providers that support tool descriptions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema describing the tool-call arguments.
    pub parameters: serde_json::Value,
    /// Provider-specific JSON extensions for tool definitions.
    #[serde(skip_serializing_if = "extensions_are_absent_or_empty")]
    pub extensions: Option<ExtraMap>,
}

impl Tool {
    /// Create a tool definition from a name and JSON Schema parameter object.
    #[must_use]
    pub fn new(name: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters,
            extensions: None,
        }
    }

    /// Attach a human-readable description for providers that surface it.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add or replace a provider-specific JSON extension field.
    #[must_use]
    pub fn with_extension(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extensions
            .get_or_insert_with(ExtraMap::new)
            .insert(key.into(), value);
        self
    }
}

/// Controls how the model selects tools for a request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolChoice {
    /// Let the model decide whether to call a tool.
    Auto,
    /// Disable tool calling for this request.
    Disabled,
    /// Require the model to call at least one tool.
    Required,
    /// Force the model to call the named tool.
    Specific {
        /// Name of the required tool.
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_serde_round_trip() {
        let tool = Tool::new("search", json!({"type": "object", "properties": {}}))
            .description("Search the web")
            .with_extension("cache", json!(true));

        let serialized = serde_json::to_string(&tool).unwrap();
        let deserialized: Tool = serde_json::from_str(&serialized).unwrap();
        assert_eq!(tool, deserialized);
    }

    #[test]
    fn tool_serde_skips_none_fields() {
        let tool = Tool::new("bare", json!({}));
        let value = serde_json::to_value(&tool).unwrap();
        let obj = value.as_object().unwrap();
        assert!(!obj.contains_key("description"));
        assert!(!obj.contains_key("extensions"));
    }

    #[test]
    fn tool_serde_skips_empty_extensions() {
        let tool = Tool {
            name: "bare".into(),
            description: None,
            parameters: json!({}),
            extensions: Some(ExtraMap::new()),
        };

        let value = serde_json::to_value(&tool).unwrap();
        let obj = value.as_object().unwrap();
        assert!(!obj.contains_key("extensions"));
    }

    #[test]
    fn tool_choice_serde_round_trip() {
        let cases = [
            ToolChoice::Auto,
            ToolChoice::Disabled,
            ToolChoice::Required,
            ToolChoice::Specific {
                name: "search".into(),
            },
        ];

        for choice in cases {
            let value = serde_json::to_value(&choice).unwrap();
            let round_tripped: ToolChoice = serde_json::from_value(value).unwrap();
            assert_eq!(round_tripped, choice);
        }
    }
}
