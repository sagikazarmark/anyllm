use std::collections::HashMap;

use anyllm::{CapabilitySupport, ChatCapability, ChatCapabilityResolver};

use crate::Model;

/// A [`ChatCapabilityResolver`] backed by `models.dev` provider data.
///
/// Answers chat capability queries from a snapshot of per-model metadata.
/// Returns `None` (defer) for models absent from the snapshot or for
/// capabilities without a clear mapping in the `models.dev` schema.
pub struct ModelsDevResolver {
    models: HashMap<String, Model>,
}

impl ModelsDevResolver {
    /// Create a resolver from a pre-built model map.
    pub fn new(models: HashMap<String, Model>) -> Self {
        Self { models }
    }

    /// Create a resolver by deserializing a single `models.dev` provider
    /// JSON entry.
    pub fn from_provider_json(json: &str) -> serde_json::Result<Self> {
        let provider: crate::Provider = serde_json::from_str(json)?;
        Ok(Self::new(provider.models))
    }
}

impl ChatCapabilityResolver for ModelsDevResolver {
    fn chat_capability(
        &self,
        model: &str,
        capability: ChatCapability,
    ) -> Option<CapabilitySupport> {
        let model = self.models.get(model)?;

        match capability {
            ChatCapability::ToolCalls => Some(model.tool_call.into()),
            ChatCapability::StructuredOutput => model.structured_output.map(Into::into),
            ChatCapability::ImageInput => {
                Some(model.modalities.input.iter().any(|m| m == "image").into())
            }
            ChatCapability::ImageOutput => {
                Some(model.modalities.output.iter().any(|m| m == "image").into())
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Limit, Modalities};

    fn test_model(
        tool_call: bool,
        structured_output: Option<bool>,
        modalities: Modalities,
    ) -> Model {
        Model {
            id: "test-model".into(),
            name: "Test Model".into(),
            family: None,
            attachment: false,
            reasoning: true,
            tool_call,
            temperature: None,
            structured_output,
            knowledge: None,
            release_date: None,
            last_updated: None,
            modalities,
            open_weights: false,
            cost: None,
            limit: Limit {
                context: 128_000,
                output: 4096,
                input: None,
            },
            status: None,
            provider: None,
            interleaved: None,
            experimental: None,
        }
    }

    fn text_only_modalities() -> Modalities {
        Modalities {
            input: vec!["text".into()],
            output: vec!["text".into()],
        }
    }

    fn vision_modalities() -> Modalities {
        Modalities {
            input: vec!["text".into(), "image".into()],
            output: vec!["text".into()],
        }
    }

    fn image_gen_modalities() -> Modalities {
        Modalities {
            input: vec!["text".into(), "image".into()],
            output: vec!["text".into(), "image".into()],
        }
    }

    #[test]
    fn tool_call_true_returns_supported() {
        let mut models = HashMap::new();
        models.insert("m".into(), test_model(true, None, text_only_modalities()));
        let resolver = ModelsDevResolver::new(models);
        assert_eq!(
            resolver.chat_capability("m", ChatCapability::ToolCalls),
            Some(CapabilitySupport::Supported),
        );
    }

    #[test]
    fn tool_call_false_returns_unsupported() {
        let mut models = HashMap::new();
        models.insert("m".into(), test_model(false, None, text_only_modalities()));
        let resolver = ModelsDevResolver::new(models);
        assert_eq!(
            resolver.chat_capability("m", ChatCapability::ToolCalls),
            Some(CapabilitySupport::Unsupported),
        );
    }

    #[test]
    fn structured_output_some_true_returns_supported() {
        let mut models = HashMap::new();
        models.insert(
            "m".into(),
            test_model(false, Some(true), text_only_modalities()),
        );
        let resolver = ModelsDevResolver::new(models);
        assert_eq!(
            resolver.chat_capability("m", ChatCapability::StructuredOutput),
            Some(CapabilitySupport::Supported),
        );
    }

    #[test]
    fn structured_output_some_false_returns_unsupported() {
        let mut models = HashMap::new();
        models.insert(
            "m".into(),
            test_model(false, Some(false), text_only_modalities()),
        );
        let resolver = ModelsDevResolver::new(models);
        assert_eq!(
            resolver.chat_capability("m", ChatCapability::StructuredOutput),
            Some(CapabilitySupport::Unsupported),
        );
    }

    #[test]
    fn structured_output_none_returns_none() {
        let mut models = HashMap::new();
        models.insert("m".into(), test_model(false, None, text_only_modalities()));
        let resolver = ModelsDevResolver::new(models);
        assert_eq!(
            resolver.chat_capability("m", ChatCapability::StructuredOutput),
            None,
        );
    }

    #[test]
    fn image_input_from_modalities() {
        let mut models = HashMap::new();
        models.insert("m".into(), test_model(false, None, vision_modalities()));
        let resolver = ModelsDevResolver::new(models);
        assert_eq!(
            resolver.chat_capability("m", ChatCapability::ImageInput),
            Some(CapabilitySupport::Supported),
        );
    }

    #[test]
    fn no_image_input_returns_unsupported() {
        let mut models = HashMap::new();
        models.insert("m".into(), test_model(false, None, text_only_modalities()));
        let resolver = ModelsDevResolver::new(models);
        assert_eq!(
            resolver.chat_capability("m", ChatCapability::ImageInput),
            Some(CapabilitySupport::Unsupported),
        );
    }

    #[test]
    fn image_output_from_modalities() {
        let mut models = HashMap::new();
        models.insert("m".into(), test_model(false, None, image_gen_modalities()));
        let resolver = ModelsDevResolver::new(models);
        assert_eq!(
            resolver.chat_capability("m", ChatCapability::ImageOutput),
            Some(CapabilitySupport::Supported),
        );
    }

    #[test]
    fn no_image_output_returns_unsupported() {
        let mut models = HashMap::new();
        models.insert("m".into(), test_model(false, None, vision_modalities()));
        let resolver = ModelsDevResolver::new(models);
        assert_eq!(
            resolver.chat_capability("m", ChatCapability::ImageOutput),
            Some(CapabilitySupport::Unsupported),
        );
    }

    #[test]
    fn unknown_model_returns_none() {
        let resolver = ModelsDevResolver::new(HashMap::new());
        assert_eq!(
            resolver.chat_capability("missing", ChatCapability::ToolCalls),
            None,
        );
    }

    #[test]
    fn reasoning_returns_none_in_v1() {
        let mut models = HashMap::new();
        models.insert("m".into(), test_model(false, None, text_only_modalities()));
        let resolver = ModelsDevResolver::new(models);
        assert_eq!(
            resolver.chat_capability("m", ChatCapability::ReasoningOutput),
            None,
        );
    }

    #[test]
    fn streaming_returns_none() {
        let mut models = HashMap::new();
        models.insert("m".into(), test_model(false, None, text_only_modalities()));
        let resolver = ModelsDevResolver::new(models);
        assert_eq!(
            resolver.chat_capability("m", ChatCapability::Streaming),
            None,
        );
    }

    #[test]
    fn from_provider_json_deserializes_and_resolves() {
        let json = r#"{
            "id": "test",
            "name": "Test Provider",
            "env": [],
            "npm": "test-sdk",
            "doc": "https://example.com",
            "models": {
                "test-model": {
                    "id": "test-model",
                    "name": "Test Model",
                    "family": null,
                    "attachment": false,
                    "reasoning": false,
                    "tool_call": true,
                    "temperature": null,
                    "structured_output": true,
                    "knowledge": null,
                    "release_date": null,
                    "last_updated": null,
                    "modalities": { "input": ["text", "image"], "output": ["text"] },
                    "open_weights": false,
                    "cost": null,
                    "limit": { "context": 128000, "output": 4096 },
                    "status": null,
                    "provider": null,
                    "interleaved": null,
                    "experimental": null
                }
            }
        }"#;

        let resolver = ModelsDevResolver::from_provider_json(json).unwrap();
        assert_eq!(
            resolver.chat_capability("test-model", ChatCapability::ToolCalls),
            Some(CapabilitySupport::Supported),
        );
        assert_eq!(
            resolver.chat_capability("test-model", ChatCapability::StructuredOutput),
            Some(CapabilitySupport::Supported),
        );
        assert_eq!(
            resolver.chat_capability("test-model", ChatCapability::ImageInput),
            Some(CapabilitySupport::Supported),
        );
        assert_eq!(
            resolver.chat_capability("test-model", ChatCapability::ImageOutput),
            Some(CapabilitySupport::Unsupported),
        );
    }
}
