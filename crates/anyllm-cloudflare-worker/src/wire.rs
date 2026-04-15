//! Wire types for the Cloudflare Workers AI native API format.
//!
//! These types match the JSON schema expected by `worker::Ai::run()` and the
//! responses it returns for text generation models. Includes `TryFrom`
//! conversions between wire types and their anyllm equivalents.

use std::cell::Cell;

use anyllm::{
    ChatRequest as AnyChatRequest, ChatResponse as AnyChatResponse, ContentBlock, FinishReason,
    Message as AnyMessage, ResponseFormat, Result, ToolResultContent, UserContent,
};
use serde::{Deserialize, Serialize};

thread_local! {
    static NEXT_SYNTHETIC_RESPONSE_ID: Cell<u64> = const { Cell::new(0) };
}

fn next_synthetic_response_id() -> u64 {
    NEXT_SYNTHETIC_RESPONSE_ID.with(|next| {
        let id = next.get();
        next.set(id + 1);
        id
    })
}

#[cfg(test)]
pub(crate) fn reset_synthetic_response_ids_for_tests() {
    NEXT_SYNTHETIC_RESPONSE_ID.with(|next| next.set(0));
}

fn reject_unsupported_request_controls(request: &AnyChatRequest) -> Result<()> {
    if request.stop.as_ref().is_some_and(|stop| !stop.is_empty()) {
        return Err(anyllm::Error::Unsupported(
            "cloudflare-worker native chat does not support stop sequences".into(),
        ));
    }

    if request.tool_choice.is_some() {
        return Err(anyllm::Error::Unsupported(
            "cloudflare-worker native chat does not support tool_choice controls".into(),
        ));
    }

    if request.reasoning.is_some() {
        return Err(anyllm::Error::Unsupported(
            "cloudflare-worker native chat does not support reasoning controls".into(),
        ));
    }

    if request.parallel_tool_calls.is_some() {
        return Err(anyllm::Error::Unsupported(
            "cloudflare-worker native chat does not support parallel_tool_calls controls".into(),
        ));
    }

    Ok(())
}

pub(crate) fn reject_unsupported_streaming_request_features(
    request: &AnyChatRequest,
) -> Result<()> {
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err(anyllm::Error::Unsupported(
            "cloudflare-worker native chat_stream does not support streamed tool calls".into(),
        ));
    }

    if request.response_format.is_some() {
        return Err(anyllm::Error::Unsupported(
            "cloudflare-worker native chat_stream does not support response_format; Workers AI JSON Mode is non-streaming".into(),
        ));
    }

    Ok(())
}

/// Request body for Cloudflare Workers AI text generation (messages-based).
#[derive(Debug, Serialize)]
pub(crate) struct ChatRequest {
    pub messages: Vec<Message>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
}

/// A message in the Cloudflare Workers AI format.
#[derive(Debug, Serialize)]
pub(crate) struct Message {
    pub role: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<MessageToolCall>>,
}

/// A tool call within an assistant message (OpenAI-style).
#[derive(Debug, Serialize)]
pub(crate) struct MessageToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: MessageToolCallFunction,
}

/// The function payload within a tool call.
#[derive(Debug, Serialize)]
pub(crate) struct MessageToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// Tool definition in the Cloudflare Workers AI format.
///
/// CF supports both flat and OpenAI-style formats. We use the OpenAI style.
#[derive(Debug, Serialize)]
pub(crate) struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

/// Function definition within a tool.
#[derive(Debug, Serialize)]
pub(crate) struct FunctionDef {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    pub parameters: serde_json::Value,
}

/// Response from Cloudflare Workers AI text generation (non-streaming).
///
/// The native API returns `{ response: "...", tool_calls: [...], usage: {...} }`.
#[derive(Debug, Deserialize)]
pub(crate) struct ChatResponse {
    /// The generated text response.
    #[serde(default)]
    pub response: Option<ResponseContent>,

    /// Tool calls requested by the model.
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Token usage information.
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ResponseContent {
    Text(String),
    Json(serde_json::Value),
}

/// A tool call in the Cloudflare response.
#[derive(Debug, Deserialize)]
pub(crate) struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Token usage from Cloudflare Workers AI.
#[derive(Debug, Deserialize)]
pub(crate) struct Usage {
    #[serde(default)]
    pub prompt_tokens: Option<u64>,

    #[serde(default)]
    pub completion_tokens: Option<u64>,

    #[serde(default)]
    pub total_tokens: Option<u64>,
}

/// A single SSE data chunk from streaming responses.
///
/// When `stream: true`, CF returns SSE events like:
/// `data: {"response":"token_text"}`
#[derive(Debug, Deserialize)]
pub(crate) struct StreamChunk {
    #[serde(default)]
    pub response: Option<String>,
}

// --- Conversions between wire types and anyllm types ---

impl TryFrom<&AnyChatRequest> for ChatRequest {
    type Error = anyllm::Error;

    fn try_from(request: &AnyChatRequest) -> Result<Self> {
        reject_unsupported_request_controls(request)?;

        // Emit req.system as role: "system" wire messages at the front, mirroring
        // the pattern used by the OpenAI-compat adapter. Empty-content prompts
        // are filtered to match the other HTTP adapters. Typed SystemOptions are
        // ignored — Cloudflare Workers AI has no per-block instruction metadata.
        let mut messages: Vec<Message> =
            Vec::with_capacity(request.system.len() + request.messages.len());
        for prompt in &request.system {
            if prompt.content.is_empty() {
                continue;
            }
            messages.push(Message {
                role: "system".to_string(),
                content: Some(prompt.content.clone()),
                tool_call_id: None,
                tool_calls: None,
            });
        }
        for msg in &request.messages {
            messages.push(Message::try_from(msg)?);
        }

        let tools = request.tools.as_ref().and_then(|tools| {
            if tools.is_empty() {
                None
            } else {
                Some(
                    tools
                        .iter()
                        .map(|t| Tool {
                            tool_type: "function".to_string(),
                            function: FunctionDef {
                                name: t.name.clone(),
                                description: t.description.clone(),
                                parameters: t.parameters.clone(),
                            },
                        })
                        .collect(),
                )
            }
        });

        let response_format = request
            .response_format
            .as_ref()
            .map(|rf| match rf {
                ResponseFormat::Text => Ok(serde_json::json!({"type": "text"})),
                ResponseFormat::Json => Ok(serde_json::json!({"type": "json_object"})),
                ResponseFormat::JsonSchema {
                    name,
                    schema,
                    strict,
                } => {
                    if name.is_some() || strict.is_some() {
                        return Err(anyllm::Error::Unsupported(
                            "cloudflare-worker native chat supports json_schema payloads, but not json_schema name/strict controls"
                                .into(),
                        ));
                    }

                    Ok(serde_json::json!({"type": "json_schema", "json_schema": schema}))
                }
                _ => Err(anyllm::Error::Unsupported(
                    "unknown response format is not supported by the Cloudflare Worker converter"
                        .into(),
                )),
            })
            .transpose()?;

        Ok(ChatRequest {
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: None, // Not in ChatRequest, would come from provider-specific options
            seed: request.seed,
            repetition_penalty: None,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stream: None,
            tools,
            response_format,
        })
    }
}

impl TryFrom<&AnyMessage> for Message {
    type Error = anyllm::Error;

    fn try_from(msg: &AnyMessage) -> Result<Self> {
        match msg {
            AnyMessage::User { content, .. } => {
                let text = match content {
                    UserContent::Text(text) => text.clone(),
                    UserContent::Parts(parts) => {
                        // Workers AI native chat input is text-only here. Refuse
                        // to silently drop multimodal or provider-specific user
                        // parts so request normalization stays faithful.
                        let mut text_parts = Vec::with_capacity(parts.len());
                        for part in parts {
                            match part {
                                anyllm::ContentPart::Text { text } => {
                                    text_parts.push(text.as_str())
                                }
                                anyllm::ContentPart::Image { .. } => {
                                    return Err(anyllm::Error::Unsupported(
                                        "cloudflare-worker native chat does not support image user content"
                                            .into(),
                                    ));
                                }
                                anyllm::ContentPart::Other { type_name, .. } => {
                                    return Err(anyllm::Error::Unsupported(format!(
                                        "cloudflare-worker native chat does not support user content part type '{type_name}'"
                                    )));
                                }
                                _ => {
                                    return Err(anyllm::Error::Unsupported(
                                        "cloudflare-worker native chat does not support this user content part"
                                            .into(),
                                    ));
                                }
                            }
                        }
                        text_parts.join("\n")
                    }
                };
                Ok(Message {
                    role: "user".to_string(),
                    content: Some(text),
                    tool_call_id: None,
                    tool_calls: None,
                })
            }
            AnyMessage::Assistant { content, .. } => {
                if content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::Image { .. }))
                {
                    return Err(anyllm::Error::Unsupported(
                        "cloudflare-worker native chat does not support assistant image replay"
                            .into(),
                    ));
                }

                if content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::Reasoning { .. }))
                {
                    return Err(anyllm::Error::Unsupported(
                        "cloudflare-worker native chat does not support assistant reasoning replay"
                            .into(),
                    ));
                }

                if let Some(type_name) = content.iter().find_map(|block| match block {
                    ContentBlock::Other { type_name, .. } => Some(type_name.as_str()),
                    _ => None,
                }) {
                    return Err(anyllm::Error::Unsupported(format!(
                        "cloudflare-worker native chat does not support assistant content block type '{type_name}'"
                    )));
                }

                let text: String = content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");

                let tool_calls: Vec<MessageToolCall> = content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                        } => Some(MessageToolCall {
                            id: id.clone(),
                            call_type: "function".to_string(),
                            function: MessageToolCallFunction {
                                name: name.clone(),
                                arguments: arguments.clone(),
                            },
                        }),
                        _ => None,
                    })
                    .collect();

                Ok(Message {
                    role: "assistant".to_string(),
                    content: if text.is_empty() { None } else { Some(text) },
                    tool_call_id: None,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                })
            }
            AnyMessage::Tool {
                tool_call_id,
                content,
                ..
            } => match content {
                ToolResultContent::Text(text) => Ok(Message {
                    role: "tool".to_string(),
                    content: Some(text.clone()),
                    tool_call_id: Some(tool_call_id.clone()),
                    tool_calls: None,
                }),
                ToolResultContent::Parts(_) => Err(anyllm::Error::Unsupported(
                    "cloudflare worker converter does not support multimodal tool result content"
                        .into(),
                )),
            },
            _ => Err(anyllm::Error::Unsupported(
                "unknown message variant is not supported by the Cloudflare Worker converter"
                    .into(),
            )),
        }
    }
}

impl TryFrom<ChatResponse> for AnyChatResponse {
    type Error = anyllm::Error;

    fn try_from(response: ChatResponse) -> Result<Self> {
        let mut content: Vec<ContentBlock> = Vec::new();
        let synthetic_response_id = next_synthetic_response_id();

        if let Some(text) = &response.response {
            let text = match text {
                ResponseContent::Text(text) => text.clone(),
                ResponseContent::Json(value) => serde_json::to_string(value).map_err(|err| {
                    anyllm::Error::serialization(format!(
                        "failed to serialize Cloudflare structured response: {err}"
                    ))
                })?,
            };

            if !text.is_empty() {
                content.push(ContentBlock::Text { text });
            }
        }

        let has_tool_calls = response
            .tool_calls
            .as_ref()
            .is_some_and(|tc| !tc.is_empty());

        if let Some(tool_calls) = &response.tool_calls {
            for (i, tc) in tool_calls.iter().enumerate() {
                content.push(ContentBlock::ToolCall {
                    id: format!("cf_tool_{synthetic_response_id}_{i}"),
                    name: tc.name.clone(),
                    arguments: serde_json::to_string(&tc.arguments)
                        .unwrap_or_else(|_| "{}".to_string()),
                });
            }
        }

        let finish_reason = if has_tool_calls {
            Some(FinishReason::ToolCalls)
        } else {
            Some(FinishReason::Stop)
        };

        let usage = response.usage.map(|u| {
            let mut usage = anyllm::Usage::new();
            usage.input_tokens = u.prompt_tokens;
            usage.output_tokens = u.completion_tokens;
            usage.total_tokens = u.total_tokens;
            usage
        });

        let mut chat_response = AnyChatResponse::new(content);
        chat_response.finish_reason = finish_reason;
        chat_response.usage = usage;
        // CF native API doesn't echo the model back.
        Ok(chat_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyllm::{
        ChatRequest as AnyChatRequest, ContentPart, ImageSource, Message as AnyMessage,
        ReasoningConfig, Tool, ToolChoice,
    };
    use serde_json::json;

    #[test]
    fn request_conversion_preserves_supported_controls() {
        let request = AnyChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
            .system("You are concise.")
            .user("Say hello")
            .temperature(0.2)
            .top_p(0.9)
            .max_tokens(64)
            .seed(42)
            .frequency_penalty(0.1)
            .presence_penalty(0.2)
            .tools(vec![
                Tool::new(
                    "lookup_weather",
                    json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }),
                )
                .description("Look up the weather."),
            ])
            .response_format(ResponseFormat::JsonSchema {
                name: None,
                schema: json!({
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"]
                }),
                strict: None,
            });

        let wire = ChatRequest::try_from(&request).unwrap();
        let value = serde_json::to_value(wire).unwrap();

        assert_eq!(value["max_tokens"], 64);
        assert_eq!(value["temperature"].as_f64(), Some(0.2_f32 as f64));
        assert_eq!(value["top_p"].as_f64(), Some(0.9_f32 as f64));
        assert_eq!(value["seed"], 42);
        assert_eq!(value["frequency_penalty"].as_f64(), Some(0.1_f32 as f64));
        assert_eq!(value["presence_penalty"].as_f64(), Some(0.2_f32 as f64));
        assert_eq!(value["tools"][0]["function"]["name"], "lookup_weather");
        assert_eq!(value["response_format"]["type"], "json_schema");
    }

    #[test]
    fn json_schema_name_and_strict_are_rejected() {
        let request = AnyChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
            .user("Hello")
            .response_format(ResponseFormat::JsonSchema {
                name: Some("answer".into()),
                schema: json!({"type": "object"}),
                strict: Some(true),
            });

        let err = ChatRequest::try_from(&request).unwrap_err();
        assert!(matches!(
            err,
            anyllm::Error::Unsupported(message)
                if message.contains("json_schema name/strict controls")
        ));
    }

    #[test]
    fn streaming_rejects_tools() {
        let request = AnyChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
            .user("Hello")
            .tools(vec![Tool::new(
                "lookup_weather",
                json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }),
            )]);

        let err = reject_unsupported_streaming_request_features(&request).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Unsupported(message) if message.contains("streamed tool calls"))
        );
    }

    #[test]
    fn streaming_rejects_response_format() {
        let request = AnyChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
            .user("Hello")
            .response_format(ResponseFormat::Json);

        let err = reject_unsupported_streaming_request_features(&request).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Unsupported(message) if message.contains("response_format"))
        );
    }

    #[test]
    fn stop_sequences_are_rejected() {
        let request = AnyChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
            .user("Hello")
            .stop(["END"]);

        let err = ChatRequest::try_from(&request).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Unsupported(message) if message.contains("stop sequences"))
        );
    }

    #[test]
    fn tool_choice_controls_are_rejected() {
        let request = AnyChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
            .user("Hello")
            .tool_choice(ToolChoice::Required);

        let err = ChatRequest::try_from(&request).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Unsupported(message) if message.contains("tool_choice"))
        );
    }

    #[test]
    fn reasoning_controls_are_rejected() {
        let request = AnyChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
            .user("Hello")
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: Some(128),
                effort: None,
            });

        let err = ChatRequest::try_from(&request).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Unsupported(message) if message.contains("reasoning controls"))
        );
    }

    #[test]
    fn parallel_tool_call_controls_are_rejected() {
        let request = AnyChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
            .user("Hello")
            .parallel_tool_calls(true);

        let err = ChatRequest::try_from(&request).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Unsupported(message) if message.contains("parallel_tool_calls"))
        );
    }

    #[test]
    fn user_multimodal_text_parts_join_with_newlines() {
        let msg = AnyMessage::user_multimodal(vec![
            ContentPart::text("First line"),
            ContentPart::text("Second line"),
        ]);

        let wire = Message::try_from(&msg).unwrap();
        assert_eq!(wire.role, "user");
        assert_eq!(wire.content.as_deref(), Some("First line\nSecond line"));
    }

    #[test]
    fn user_image_part_is_rejected_instead_of_silently_dropped() {
        let msg = AnyMessage::user_multimodal(vec![
            ContentPart::text("Describe this"),
            ContentPart::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png".into(),
                },
                detail: None,
            },
        ]);

        let err = Message::try_from(&msg).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Unsupported(message) if message.contains("image user content"))
        );
    }

    #[test]
    fn user_other_part_is_rejected_instead_of_silently_dropped() {
        let msg = AnyMessage::user_multimodal(vec![ContentPart::Other {
            type_name: "audio".into(),
            data: serde_json::Map::new(),
        }]);

        let err = Message::try_from(&msg).unwrap_err();
        assert!(matches!(err, anyllm::Error::Unsupported(message) if message.contains("audio")));
    }

    #[test]
    fn assistant_image_replay_is_rejected_instead_of_silently_dropped() {
        let msg = AnyMessage::Assistant {
            content: vec![anyllm::ContentBlock::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png".into(),
                },
            }],
            name: None,
            extensions: None,
        };

        let err = Message::try_from(&msg).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Unsupported(message) if message.contains("assistant image replay"))
        );
    }

    #[test]
    fn assistant_reasoning_replay_is_rejected_instead_of_silently_dropped() {
        let msg = AnyMessage::Assistant {
            content: vec![anyllm::ContentBlock::Reasoning {
                text: "thinking".into(),
                signature: None,
            }],
            name: None,
            extensions: None,
        };

        let err = Message::try_from(&msg).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Unsupported(message) if message.contains("assistant reasoning replay"))
        );
    }

    #[test]
    fn assistant_other_block_is_rejected_instead_of_silently_dropped() {
        let msg = AnyMessage::Assistant {
            content: vec![anyllm::ContentBlock::Other {
                type_name: "audio".into(),
                data: serde_json::Map::new(),
            }],
            name: None,
            extensions: None,
        };

        let err = Message::try_from(&msg).unwrap_err();
        assert!(matches!(err, anyllm::Error::Unsupported(message) if message.contains("audio")));
    }

    #[test]
    fn synthetic_tool_call_ids_are_unique_per_response_conversion() {
        reset_synthetic_response_ids_for_tests();

        let make_response = || ChatResponse {
            response: None,
            tool_calls: Some(vec![ToolCall {
                name: "lookup_weather".into(),
                arguments: serde_json::json!({"city": "London"}),
            }]),
            usage: None,
        };

        let first = anyllm::ChatResponse::try_from(make_response()).unwrap();
        let second = anyllm::ChatResponse::try_from(make_response()).unwrap();

        let first_id = first.tool_calls().next().unwrap().id.to_string();
        let second_id = second.tool_calls().next().unwrap().id.to_string();

        assert_ne!(first_id, second_id);
        assert!(first_id.starts_with("cf_tool_"));
        assert!(second_id.starts_with("cf_tool_"));
    }

    #[test]
    fn tool_call_response_sets_tool_calls_finish_reason() {
        reset_synthetic_response_ids_for_tests();

        let response = ChatResponse {
            response: None,
            tool_calls: Some(vec![ToolCall {
                name: "lookup_weather".into(),
                arguments: json!({"city": "London"}),
            }]),
            usage: None,
        };

        let response = anyllm::ChatResponse::try_from(response).unwrap();
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        let tool_call = response.tool_calls().next().unwrap();
        assert_eq!(tool_call.name, "lookup_weather");
        assert_eq!(tool_call.arguments, r#"{"city":"London"}"#);
    }

    #[test]
    fn structured_json_response_is_serialized_into_text_content() {
        let response = ChatResponse {
            response: Some(ResponseContent::Json(json!({
                "token": "cf-json-ok",
                "summary": "Cloudflare JSON mode works."
            }))),
            tool_calls: None,
            usage: None,
        };

        let response = anyllm::ChatResponse::try_from(response).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&response.text_or_empty()).unwrap();
        assert_eq!(
            parsed,
            json!({
                "token": "cf-json-ok",
                "summary": "Cloudflare JSON mode works."
            })
        );
    }
}
