use anyllm::{ChatRequest, ChatResponse, ExtraMap, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub(crate) struct CreateMessageRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExtraMap>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemBlock>>,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Message {
    pub role: String,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ImageSource {
    Url { url: String },
    Base64 { media_type: String, data: String },
}

#[derive(Debug, Serialize)]
pub(crate) struct ToolDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

#[derive(Debug, Serialize)]
pub(crate) struct ThinkingConfig {
    #[serde(rename = "type")]
    pub config_type: String,
    pub budget_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CreateMessageResponse {
    pub id: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ErrorResponse {
    pub error: Error,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Error {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// --- Conversions between wire types and anyllm types ---

use anyllm::{ToolResultContent, UserContent};

fn into_anyllm_image_source(source: ImageSource) -> anyllm::ImageSource {
    match source {
        ImageSource::Url { url } => anyllm::ImageSource::Url { url },
        ImageSource::Base64 { media_type, data } => {
            anyllm::ImageSource::Base64 { media_type, data }
        }
    }
}

/// Default max_tokens if not specified in the request.
const DEFAULT_MAX_TOKENS: u32 = 4096;

fn ensure_reasoning_budget_fits_max_tokens(effective_max_tokens: u32, budget: u32) -> Result<u32> {
    if budget < effective_max_tokens {
        return Ok(effective_max_tokens);
    }

    budget.checked_add(1).ok_or_else(|| {
        anyllm::Error::InvalidRequest("anthropic reasoning budget_tokens cannot be u32::MAX".into())
    })
}

impl TryFrom<&ChatRequest> for CreateMessageRequest {
    type Error = anyllm::Error;

    fn try_from(request: &ChatRequest) -> Result<Self> {
        let provider_options = request.option::<crate::ChatRequestOptions>();

        match &request.response_format {
            None | Some(anyllm::ResponseFormat::Text) => {}
            Some(_) => {
                return Err(anyllm::Error::Unsupported(
                    "anthropic does not support response_format; use plain text or anyllm extraction fallback"
                        .into(),
                ));
            }
        }

        if request.parallel_tool_calls.is_some() {
            return Err(anyllm::Error::Unsupported(
                "anthropic does not support parallel_tool_calls controls".into(),
            ));
        }
        if request.seed.is_some() {
            return Err(anyllm::Error::Unsupported(
                "anthropic does not support seed controls".into(),
            ));
        }
        if request.frequency_penalty.is_some() {
            return Err(anyllm::Error::Unsupported(
                "anthropic does not support frequency_penalty controls".into(),
            ));
        }
        if request.presence_penalty.is_some() {
            return Err(anyllm::Error::Unsupported(
                "anthropic does not support presence_penalty controls".into(),
            ));
        }

        // Extract system messages and hoist to top-level
        let mut system_blocks: Vec<SystemBlock> = Vec::new();
        let mut api_messages: Vec<Message> = Vec::with_capacity(request.messages.len());
        let mut pending_tool_results: Vec<ContentBlock> = Vec::new();

        let flush_tool_results =
            |api_messages: &mut Vec<Message>, pending_tool_results: &mut Vec<ContentBlock>| {
                if !pending_tool_results.is_empty() {
                    api_messages.push(Message {
                        role: "user".to_string(),
                        content: std::mem::take(pending_tool_results),
                    });
                }
            };

        // Emit req.system first. Order is preserved.
        for prompt in &request.system {
            if prompt.content.is_empty() {
                continue;
            }
            let mut block = SystemBlock {
                block_type: "text".to_string(),
                text: prompt.content.clone(),
                cache_control: None,
            };
            if let Some(cc) = prompt.option::<crate::CacheControl>() {
                block.cache_control = Some(cc.to_wire());
            }
            system_blocks.push(block);
        }

        for msg in &request.messages {
            match msg {
                anyllm::Message::User { content, .. } => {
                    flush_tool_results(&mut api_messages, &mut pending_tool_results);
                    let api_content = match content {
                        UserContent::Text(text) => {
                            vec![ContentBlock::Text { text: text.clone() }]
                        }
                        UserContent::Parts(parts) => {
                            let mut api_content = Vec::with_capacity(parts.len());
                            for part in parts {
                                let block = match part {
                                    anyllm::ContentPart::Text { text } => {
                                        ContentBlock::Text { text: text.clone() }
                                    }
                                    anyllm::ContentPart::Image { source, .. } => {
                                        let source = match source {
                                            anyllm::ImageSource::Url { url } => {
                                                ImageSource::Url { url: url.clone() }
                                            }
                                            anyllm::ImageSource::Base64 { media_type, data } => {
                                                ImageSource::Base64 {
                                                    media_type: media_type.clone(),
                                                    data: data.clone(),
                                                }
                                            }
                                            _ => {
                                                return Err(anyllm::Error::Unsupported(
                                                    "unknown image source is not supported by the anthropic converter"
                                                        .into(),
                                                ));
                                            }
                                        };
                                        ContentBlock::Image { source }
                                    }
                                    anyllm::ContentPart::Other { type_name, .. } => {
                                        return Err(anyllm::Error::Unsupported(format!(
                                            "anthropic does not support user content part type '{type_name}'"
                                        )));
                                    }
                                    _ => {
                                        return Err(anyllm::Error::Unsupported(
                                            "unsupported user content part for anthropic conversion"
                                                .into(),
                                        ));
                                    }
                                };
                                api_content.push(block);
                            }
                            api_content
                        }
                    };
                    api_messages.push(Message {
                        role: "user".to_string(),
                        content: api_content,
                    });
                }
                anyllm::Message::Assistant { content, .. } => {
                    flush_tool_results(&mut api_messages, &mut pending_tool_results);
                    let mut api_content: Vec<ContentBlock> = Vec::with_capacity(content.len());
                    for block in content {
                        let converted = match block {
                            anyllm::ContentBlock::Text { text } => {
                                Some(ContentBlock::Text { text: text.clone() })
                            }
                            anyllm::ContentBlock::ToolCall {
                                id,
                                name,
                                arguments,
                            } => {
                                let input: serde_json::Value = serde_json::from_str(arguments)
                                    .map_err(|e| {
                                        anyllm::Error::serialization(format!(
                                            "invalid JSON in tool call '{}' arguments: {}",
                                            name, e
                                        ))
                                    })?;
                                Some(ContentBlock::ToolUse {
                                    id: id.clone(),
                                    name: name.clone(),
                                    input,
                                })
                            }
                            anyllm::ContentBlock::Reasoning { text, signature } => {
                                Some(ContentBlock::Thinking {
                                    thinking: text.clone(),
                                    signature: signature.clone(),
                                })
                            }
                            anyllm::ContentBlock::Image { .. } => {
                                return Err(anyllm::Error::Unsupported(
                                    "anthropic does not support assistant image replay".into(),
                                ));
                            }
                            anyllm::ContentBlock::Other { .. } | _ => None,
                        };

                        if let Some(converted) = converted {
                            api_content.push(converted);
                        }
                    }
                    api_messages.push(Message {
                        role: "assistant".to_string(),
                        content: api_content,
                    });
                }
                anyllm::Message::Tool {
                    tool_call_id,
                    content,
                    is_error,
                    ..
                } => {
                    let content = match content {
                        ToolResultContent::Text(text) => text.clone(),
                        ToolResultContent::Parts(_) => {
                            return Err(anyllm::Error::Unsupported(
                                "anthropic does not support multimodal tool result replay".into(),
                            ));
                        }
                    };
                    pending_tool_results.push(ContentBlock::ToolResult {
                        tool_use_id: tool_call_id.clone(),
                        content,
                        is_error: is_error.unwrap_or(false),
                    });
                }
                _ => {
                    return Err(anyllm::Error::Unsupported(
                        "unknown message variant is not supported by the anthropic converter"
                            .into(),
                    ));
                }
            }
        }

        flush_tool_results(&mut api_messages, &mut pending_tool_results);

        // Merge consecutive messages with the same role.
        //
        // The Anthropic API requires strictly alternating user/assistant roles.
        // Summary messages from compaction are injected as user-role messages
        // (the universal role choice per spec), which can create consecutive
        // user messages. This pass merges them by concatenating content blocks.
        api_messages = merge_consecutive_same_role(api_messages);

        let system = if system_blocks.is_empty() {
            None
        } else {
            Some(system_blocks)
        };

        // Map tools — omit entirely if ToolChoice::Disabled
        let is_disabled = matches!(request.tool_choice, Some(anyllm::ToolChoice::Disabled));
        let tools = if is_disabled {
            None
        } else {
            request.tools.as_ref().and_then(|tools| {
                if tools.is_empty() {
                    None
                } else {
                    Some(
                        tools
                            .iter()
                            .map(|t| ToolDefinition {
                                name: t.name.clone(),
                                description: t.description.clone(),
                                input_schema: t.parameters.clone(),
                            })
                            .collect(),
                    )
                }
            })
        };

        // Map tool choice — skip if Disabled (tools omitted)
        let tool_choice = if is_disabled {
            None
        } else {
            request
                .tool_choice
                .as_ref()
                .map(|tc| match tc {
                    anyllm::ToolChoice::Auto => Ok(ToolChoice::Auto),
                    anyllm::ToolChoice::Required => Ok(ToolChoice::Any),
                    anyllm::ToolChoice::Specific { name } => {
                        Ok(ToolChoice::Tool { name: name.clone() })
                    }
                    anyllm::ToolChoice::Disabled => unreachable!(),
                    _ => Err(anyllm::Error::Unsupported(
                        "unknown tool choice is not supported by the anthropic converter".into(),
                    )),
                })
                .transpose()?
        };

        // Determine effective max_tokens first so the thinking budget can reference it.
        let mut effective_max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);

        // Map reasoning config.
        // Provider-specific budget_tokens overrides ChatRequest.reasoning.
        // The Anthropic API requires budget_tokens < max_tokens. When both default
        // to the same value we bump max_tokens up to ensure the constraint holds.
        let provider_budget = provider_options.and_then(|opts| opts.budget_tokens);
        let thinking = if let Some(budget) = provider_budget {
            // Provider-specific override — always enable thinking with this budget.
            if budget >= effective_max_tokens {
                effective_max_tokens =
                    ensure_reasoning_budget_fits_max_tokens(effective_max_tokens, budget)?;
            }
            Some(ThinkingConfig {
                config_type: "enabled".to_string(),
                budget_tokens: budget,
            })
        } else if let Some(rc) = request.reasoning.as_ref() {
            if rc.enabled {
                let budget = rc.budget_tokens.unwrap_or({
                    if matches!(rc.effort, Some(anyllm::ReasoningEffort::Low)) {
                        1024
                    } else if matches!(rc.effort, Some(anyllm::ReasoningEffort::Medium)) {
                        4096
                    } else {
                        DEFAULT_MAX_TOKENS
                    }
                });
                if budget >= effective_max_tokens {
                    effective_max_tokens =
                        ensure_reasoning_budget_fits_max_tokens(effective_max_tokens, budget)?;
                }
                Some(ThinkingConfig {
                    config_type: "enabled".to_string(),
                    budget_tokens: budget,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Map stop sequences
        let stop_sequences = request.stop.as_ref().and_then(|seqs| {
            if seqs.is_empty() {
                None
            } else {
                Some(seqs.clone())
            }
        });

        Ok(CreateMessageRequest {
            model: request.model.clone(),
            max_tokens: effective_max_tokens,
            metadata: provider_options.and_then(|opts| opts.metadata.clone()),
            system,
            messages: api_messages,
            tools,
            tool_choice,
            temperature: request.temperature,
            top_p: request.top_p,
            stop_sequences,
            thinking,
            top_k: provider_options.and_then(|opts| opts.top_k),
            stream: false,
        })
    }
}

impl TryFrom<CreateMessageResponse> for ChatResponse {
    type Error = anyllm::Error;

    fn try_from(response: CreateMessageResponse) -> Result<Self> {
        let content: Vec<anyllm::ContentBlock> = response
            .content
            .into_iter()
            .map(|block| match block {
                ContentBlock::Text { text } => Ok(anyllm::ContentBlock::Text { text }),
                ContentBlock::Image { source } => Ok(anyllm::ContentBlock::Image {
                    source: into_anyllm_image_source(source),
                }),
                ContentBlock::ToolUse { id, name, input } => {
                    let arguments = serde_json::to_string(&input).map_err(anyllm::Error::from)?;
                    Ok(anyllm::ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    })
                }
                ContentBlock::ToolResult { .. } => Ok(anyllm::ContentBlock::Text {
                    text: String::new(),
                }),
                ContentBlock::Thinking {
                    thinking,
                    signature,
                } => Ok(anyllm::ContentBlock::Reasoning {
                    text: thinking,
                    signature,
                }),
            })
            .collect::<Result<Vec<_>>>()?;

        let finish_reason = response.stop_reason.as_deref().map(parse_stop_reason);

        let mut usage = anyllm::Usage::new()
            .input_tokens(response.usage.input_tokens)
            .output_tokens(response.usage.output_tokens);
        usage.cached_input_tokens = response.usage.cache_read_input_tokens;
        usage.cache_creation_input_tokens = response.usage.cache_creation_input_tokens;

        let mut chat_response = anyllm::ChatResponse::new(content);
        chat_response.finish_reason = finish_reason;
        chat_response.usage = Some(usage);
        chat_response.model = Some(response.model);
        chat_response.id = Some(response.id);
        Ok(chat_response)
    }
}

/// Parse an Anthropic stop_reason string to an anyllm FinishReason.
pub(crate) fn parse_stop_reason(s: &str) -> anyllm::FinishReason {
    match s {
        "end_turn" | "stop_sequence" => anyllm::FinishReason::Stop,
        "max_tokens" => anyllm::FinishReason::Length,
        "tool_use" => anyllm::FinishReason::ToolCalls,
        other => anyllm::FinishReason::Other(other.to_string()),
    }
}

/// Merge consecutive `Message`s that share the same role.
///
/// When two adjacent messages have the same role, the second message's
/// content blocks are appended to the first and the second is removed.
/// This ensures the Anthropic API's alternating-role invariant is met
/// even when compaction summaries create adjacent user-role messages.
fn merge_consecutive_same_role(messages: Vec<Message>) -> Vec<Message> {
    let mut merged: Vec<Message> = Vec::with_capacity(messages.len());
    for msg in messages {
        if let Some(last) = merged.last_mut()
            && last.role == msg.role
        {
            last.content.extend(msg.content);
            continue;
        }
        merged.push(msg);
    }
    merged
}

#[cfg(test)]
pub(crate) fn conformance_to_api_request(
    request: &anyllm::ChatRequest,
    stream: bool,
) -> Result<serde_json::Value> {
    let mut req = CreateMessageRequest::try_from(request)?;
    req.stream = stream;
    serde_json::to_value(req).map_err(anyllm::Error::from)
}

#[cfg(test)]
pub(crate) fn conformance_from_api_response_value(
    value: serde_json::Value,
) -> Result<ChatResponse> {
    let response: CreateMessageResponse =
        serde_json::from_value(value).map_err(anyllm::Error::from)?;
    response.try_into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serialize_minimal_request() {
        let req = CreateMessageRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 1024,
            metadata: None,
            system: None,
            messages: vec![Message {
                role: "user".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Hello!".to_string(),
                }],
            }],
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            stop_sequences: None,
            thinking: None,
            top_k: None,
            stream: true,
        };

        let value = serde_json::to_value(&req).unwrap();

        assert_eq!(value["model"], "claude-sonnet-4-20250514");
        assert_eq!(value["max_tokens"], 1024);
        assert_eq!(value["stream"], true);
        assert_eq!(value["messages"][0]["role"], "user");
        assert_eq!(value["messages"][0]["content"][0]["type"], "text");
        assert_eq!(value["messages"][0]["content"][0]["text"], "Hello!");
        // Optional fields that are None should be absent
        assert!(value.get("system").is_none());
        assert!(value.get("tools").is_none());
        assert!(value.get("tool_choice").is_none());
        assert!(value.get("temperature").is_none());
        assert!(value.get("top_p").is_none());
        assert!(value.get("stop_sequences").is_none());
        assert!(value.get("thinking").is_none());
    }

    #[test]
    fn serialize_full_request() {
        let req = CreateMessageRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 4096,
            metadata: Some(serde_json::Map::from_iter([(
                "trace_id".to_string(),
                json!("trace-anthropic-123"),
            )])),
            system: Some(vec![SystemBlock {
                block_type: "text".to_string(),
                text: "You are a helpful assistant.".to_string(),
                cache_control: Some(json!({"type": "ephemeral"})),
            }]),
            messages: vec![Message {
                role: "user".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Hello!".to_string(),
                }],
            }],
            tools: Some(vec![ToolDefinition {
                name: "read_file".to_string(),
                description: Some("Read a file".to_string()),
                input_schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            }]),
            tool_choice: Some(ToolChoice::Auto),
            temperature: Some(0.75),
            top_p: Some(0.5),
            stop_sequences: Some(vec!["END".to_string()]),
            thinking: Some(ThinkingConfig {
                config_type: "enabled".to_string(),
                budget_tokens: 2048,
            }),
            top_k: Some(25),
            stream: false,
        };

        let value = serde_json::to_value(&req).unwrap();

        assert_eq!(value["model"], "claude-sonnet-4-20250514");
        assert_eq!(value["max_tokens"], 4096);
        assert_eq!(value["stream"], false);
        assert_eq!(value["temperature"], 0.75);
        assert_eq!(value["top_p"], 0.5);
        assert_eq!(value["top_k"], 25);
        assert_eq!(
            value["metadata"],
            json!({"trace_id": "trace-anthropic-123"})
        );
        assert_eq!(value["stop_sequences"][0], "END");

        // System blocks
        assert_eq!(value["system"][0]["type"], "text");
        assert_eq!(value["system"][0]["text"], "You are a helpful assistant.");
        assert_eq!(
            value["system"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );

        // Tools
        assert_eq!(value["tools"][0]["name"], "read_file");
        assert_eq!(value["tools"][0]["description"], "Read a file");

        // Tool choice
        assert_eq!(value["tool_choice"]["type"], "auto");

        // Thinking
        assert_eq!(value["thinking"]["type"], "enabled");
        assert_eq!(value["thinking"]["budget_tokens"], 2048);
    }

    #[test]
    fn deserialize_text_response() {
        let json_str = r#"{
            "id": "msg_01234",
            "content": [{"type": "text", "text": "Hello, world!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        }"#;

        let resp: CreateMessageResponse = serde_json::from_str(json_str).unwrap();

        assert_eq!(resp.id, "msg_01234");
        assert_eq!(resp.model, "claude-sonnet-4-20250514");
        assert_eq!(resp.stop_reason, Some("end_turn".to_string()));
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        assert!(resp.usage.cache_read_input_tokens.is_none());
        assert!(resp.usage.cache_creation_input_tokens.is_none());
        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Hello, world!"),
            other => panic!("Expected Text block, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_response_with_null_stop_reason() {
        let json_str = r#"{
            "id": "msg_01234",
            "content": [{"type": "text", "text": "Streaming..."}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        }"#;

        let resp: CreateMessageResponse = serde_json::from_str(json_str).unwrap();
        assert!(resp.stop_reason.is_none());
    }

    #[test]
    fn deserialize_tool_use_response() {
        let json_str = r#"{
            "id": "msg_56789",
            "content": [
                {"type": "text", "text": "Let me read that file."},
                {"type": "tool_use", "id": "toolu_xyz", "name": "read_file", "input": {"path": "/etc/hosts"}}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 50,
                "output_tokens": 30
            }
        }"#;

        let resp: CreateMessageResponse = serde_json::from_str(json_str).unwrap();

        assert_eq!(resp.id, "msg_56789");
        assert_eq!(resp.stop_reason, Some("tool_use".to_string()));
        assert_eq!(resp.content.len(), 2);

        match &resp.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Let me read that file."),
            other => panic!("Expected Text block, got {other:?}"),
        }
        match &resp.content[1] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_xyz");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "/etc/hosts");
            }
            other => panic!("Expected ToolUse block, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_thinking_response() {
        let json_str = r#"{
            "id": "msg_think",
            "content": [
                {"type": "thinking", "thinking": "Let me reason about this...", "signature": "sig_abc"},
                {"type": "text", "text": "Here is my answer."}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 100
            }
        }"#;

        let resp: CreateMessageResponse = serde_json::from_str(json_str).unwrap();
        assert_eq!(resp.content.len(), 2);

        match &resp.content[0] {
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                assert_eq!(thinking, "Let me reason about this...");
                assert_eq!(signature, &Some("sig_abc".to_string()));
            }
            other => panic!("Expected Thinking block, got {other:?}"),
        }
        match &resp.content[1] {
            ContentBlock::Text { text } => assert_eq!(text, "Here is my answer."),
            other => panic!("Expected Text block, got {other:?}"),
        }
    }

    #[test]
    fn deserialize_response_with_cache_usage() {
        let json_str = r#"{
            "id": "msg_cache",
            "content": [{"type": "text", "text": "Cached response."}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 15
            }
        }"#;

        let resp: CreateMessageResponse = serde_json::from_str(json_str).unwrap();

        assert_eq!(resp.usage.input_tokens, 100);
        assert_eq!(resp.usage.output_tokens, 20);
        assert_eq!(resp.usage.cache_read_input_tokens, Some(80));
        assert_eq!(resp.usage.cache_creation_input_tokens, Some(15));
    }

    // --- Conversion tests ---

    use crate::ChatRequestOptions;
    use anyllm::{ContentPart, ReasoningConfig, Tool};

    fn try_from_request(request: &ChatRequest) -> Result<CreateMessageRequest> {
        CreateMessageRequest::try_from(request)
    }

    fn try_into_response(response: CreateMessageResponse) -> Result<ChatResponse> {
        response.try_into()
    }

    #[test]
    fn converts_simple_user_message() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .message(anyllm::Message::user("Hello, Claude!"));

        let api_req = try_from_request(&req).unwrap();

        assert_eq!(api_req.model, "claude-sonnet-4-20250514");
        assert_eq!(api_req.max_tokens, DEFAULT_MAX_TOKENS);
        assert!(!api_req.stream);
        assert!(api_req.system.is_none());
        assert_eq!(api_req.messages.len(), 1);
        assert_eq!(api_req.messages[0].role, "user");
        match &api_req.messages[0].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Hello, Claude!"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn hoists_system_messages_to_top_level() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .system("You are helpful")
            .message(anyllm::Message::user("Hi"));

        let api_req = try_from_request(&req).unwrap();

        let system = api_req.system.unwrap();
        assert_eq!(system.len(), 1);
        assert_eq!(system[0].text, "You are helpful");
        assert_eq!(system[0].block_type, "text");
        assert!(system[0].cache_control.is_none());
        assert_eq!(api_req.messages.len(), 1);
        assert_eq!(api_req.messages[0].role, "user");
    }

    #[test]
    fn hoists_multiple_system_messages() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .system("First")
            .system("Second")
            .message(anyllm::Message::user("Hi"));

        let api_req = try_from_request(&req).unwrap();

        let system = api_req.system.unwrap();
        assert_eq!(system.len(), 2);
        assert_eq!(system[0].text, "First");
        assert_eq!(system[1].text, "Second");
        assert_eq!(api_req.messages.len(), 1);
    }

    #[test]
    fn system_message_with_cache_control() {
        use crate::CacheControl;
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514").system(
            anyllm::SystemPrompt::new("Cached system").with_option(CacheControl::ephemeral()),
        );

        let api_req = try_from_request(&req).unwrap();

        let system = api_req.system.unwrap();
        assert_eq!(system[0].cache_control, Some(json!({"type": "ephemeral"})));
    }

    #[test]
    fn converts_assistant_message_with_tool_calls() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514").message(
            anyllm::Message::Assistant {
                content: vec![
                    anyllm::ContentBlock::Text {
                        text: "Let me check.".into(),
                    },
                    anyllm::ContentBlock::ToolCall {
                        id: "toolu_abc".into(),
                        name: "read_file".into(),
                        arguments: r#"{"path":"/tmp/test.txt"}"#.into(),
                    },
                ],
                name: None,
                extensions: None,
            },
        );

        let api_req = try_from_request(&req).unwrap();

        assert_eq!(api_req.messages[0].role, "assistant");
        assert_eq!(api_req.messages[0].content.len(), 2);
        match &api_req.messages[0].content[1] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_abc");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "/tmp/test.txt");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn converts_assistant_message_with_reasoning() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514").message(
            anyllm::Message::Assistant {
                content: vec![
                    anyllm::ContentBlock::Reasoning {
                        text: "Let me think...".into(),
                        signature: Some("sig_abc".into()),
                    },
                    anyllm::ContentBlock::Text {
                        text: "Answer".into(),
                    },
                ],
                name: None,
                extensions: None,
            },
        );

        let api_req = try_from_request(&req).unwrap();

        match &api_req.messages[0].content[0] {
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                assert_eq!(thinking, "Let me think...");
                assert_eq!(signature, &Some("sig_abc".to_string()));
            }
            other => panic!("expected Thinking, got {other:?}"),
        }
    }

    #[test]
    fn provider_options_set_anthropic_top_k() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .with_option(ChatRequestOptions {
                top_k: Some(25),
                metadata: None,
                anthropic_beta: Vec::new(),
                budget_tokens: None,
            })
            .message(anyllm::Message::user("Hello"));

        let api_req = try_from_request(&req).unwrap();
        assert_eq!(api_req.top_k, Some(25));
    }

    #[test]
    fn converts_tool_result_to_user_role() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514").message(
            anyllm::Message::tool_result("toolu_abc", "read_file", "file contents here"),
        );

        let api_req = try_from_request(&req).unwrap();

        assert_eq!(api_req.messages[0].role, "user");
        match &api_req.messages[0].content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "toolu_abc");
                assert_eq!(content, "file contents here");
                assert!(!is_error);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn converts_tool_error_message() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514").message(
            anyllm::Message::tool_error("toolu_abc", "read_file", "Permission denied"),
        );

        let api_req = try_from_request(&req).unwrap();

        match &api_req.messages[0].content[0] {
            ContentBlock::ToolResult { is_error, .. } => {
                assert!(is_error);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn groups_consecutive_tool_results_into_one_user_message() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .message(anyllm::Message::tool_result(
                "toolu_1",
                "tool_a",
                "result one",
            ))
            .message(anyllm::Message::tool_error(
                "toolu_2",
                "tool_b",
                "result two",
            ));

        let api_req = try_from_request(&req).unwrap();

        assert_eq!(api_req.messages.len(), 1);
        assert_eq!(api_req.messages[0].role, "user");
        assert_eq!(api_req.messages[0].content.len(), 2);
        match &api_req.messages[0].content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "toolu_1");
                assert_eq!(content, "result one");
                assert!(!is_error);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
        match &api_req.messages[0].content[1] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "toolu_2");
                assert_eq!(content, "result two");
                assert!(*is_error);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn flushes_tool_result_group_at_non_tool_boundaries() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .message(anyllm::Message::tool_result("toolu_1", "tool_a", "before"))
            .message(anyllm::Message::assistant("done with tools"))
            .message(anyllm::Message::tool_result("toolu_2", "tool_b", "after"));

        let api_req = try_from_request(&req).unwrap();

        assert_eq!(api_req.messages.len(), 3);
        assert_eq!(api_req.messages[0].role, "user");
        assert_eq!(api_req.messages[1].role, "assistant");
        assert_eq!(api_req.messages[2].role, "user");
        assert_eq!(api_req.messages[0].content.len(), 1);
        assert_eq!(api_req.messages[2].content.len(), 1);
    }

    #[test]
    fn merges_consecutive_user_messages() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .message(anyllm::Message::user("summary of earlier conversation"))
            .message(anyllm::Message::user("actual user question"));

        let api_req = try_from_request(&req).unwrap();

        assert_eq!(api_req.messages.len(), 1);
        assert_eq!(api_req.messages[0].role, "user");
        assert_eq!(api_req.messages[0].content.len(), 2);
        match &api_req.messages[0].content[0] {
            ContentBlock::Text { text } => {
                assert_eq!(text, "summary of earlier conversation")
            }
            other => panic!("expected Text, got {other:?}"),
        }
        match &api_req.messages[0].content[1] {
            ContentBlock::Text { text } => assert_eq!(text, "actual user question"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn does_not_merge_alternating_roles() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .message(anyllm::Message::user("q1"))
            .message(anyllm::Message::assistant("a1"))
            .message(anyllm::Message::user("q2"));

        let api_req = try_from_request(&req).unwrap();

        assert_eq!(api_req.messages.len(), 3);
        assert_eq!(api_req.messages[0].role, "user");
        assert_eq!(api_req.messages[1].role, "assistant");
        assert_eq!(api_req.messages[2].role, "user");
    }

    #[test]
    fn merges_user_after_tool_result_flush() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .message(anyllm::Message::tool_result("toolu_1", "tool_a", "result"))
            .message(anyllm::Message::user("follow-up question"));

        let api_req = try_from_request(&req).unwrap();

        assert_eq!(api_req.messages.len(), 1);
        assert_eq!(api_req.messages[0].role, "user");
        assert_eq!(api_req.messages[0].content.len(), 2);
    }

    #[test]
    fn maps_tools_to_api_definitions() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .tools(vec![
                Tool::new(
                    "read_file",
                    json!({"type": "object", "properties": {"path": {"type": "string"}}}),
                )
                .description("Read a file"),
            ])
            .tool_choice(anyllm::ToolChoice::Auto)
            .message(anyllm::Message::user("Hi"));

        let api_req = try_from_request(&req).unwrap();

        let tools = api_req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[0].description, Some("Read a file".to_string()));

        match api_req.tool_choice.unwrap() {
            ToolChoice::Auto => {}
            other => panic!("expected Auto, got {other:?}"),
        }
    }

    #[test]
    fn tool_choice_required_maps_to_any() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .tools(vec![Tool::new("t", json!({})).description("d")])
            .tool_choice(anyllm::ToolChoice::Required)
            .message(anyllm::Message::user("Hi"));

        let api_req = try_from_request(&req).unwrap();
        assert!(matches!(api_req.tool_choice, Some(ToolChoice::Any)));
    }

    #[test]
    fn tool_choice_specific_maps_to_tool() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .tools(vec![Tool::new("read_file", json!({})).description("d")])
            .tool_choice(anyllm::ToolChoice::Specific {
                name: "read_file".into(),
            })
            .message(anyllm::Message::user("Hi"));

        let api_req = try_from_request(&req).unwrap();
        match api_req.tool_choice.unwrap() {
            ToolChoice::Tool { name } => assert_eq!(name, "read_file"),
            other => panic!("expected Tool, got {other:?}"),
        }
    }

    #[test]
    fn tool_choice_disabled_omits_tools() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .tools(vec![Tool::new("read_file", json!({})).description("d")])
            .tool_choice(anyllm::ToolChoice::Disabled)
            .message(anyllm::Message::user("Hi"));

        let api_req = try_from_request(&req).unwrap();
        assert!(api_req.tools.is_none());
        assert!(api_req.tool_choice.is_none());
    }

    #[test]
    fn empty_tools_vec_maps_to_none() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .tools(vec![])
            .message(anyllm::Message::user("Hi"));

        let api_req = try_from_request(&req).unwrap();
        assert!(api_req.tools.is_none());
    }

    #[test]
    fn maps_optional_fields() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .temperature(0.7)
            .top_p(0.9)
            .max_tokens(8192)
            .stop(["END"])
            .message(anyllm::Message::user("Hi"));

        let mut api_req = try_from_request(&req).unwrap();
        api_req.stream = true;

        assert_eq!(api_req.temperature, Some(0.7));
        assert_eq!(api_req.top_p, Some(0.9));
        assert_eq!(api_req.max_tokens, 8192);
        assert_eq!(api_req.stop_sequences, Some(vec!["END".to_string()]));
        assert!(api_req.stream);
    }

    #[test]
    fn empty_stop_sequences_maps_to_none() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .stop(std::iter::empty::<&str>())
            .message(anyllm::Message::user("Hi"));

        let api_req = try_from_request(&req).unwrap();
        assert!(api_req.stop_sequences.is_none());
    }

    #[test]
    fn maps_reasoning_config_enabled() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: Some(2048),
                effort: None,
            })
            .message(anyllm::Message::user("Hi"));

        let api_req = try_from_request(&req).unwrap();
        let thinking = api_req.thinking.unwrap();
        assert_eq!(thinking.config_type, "enabled");
        assert_eq!(thinking.budget_tokens, 2048);
    }

    #[test]
    fn reasoning_disabled_maps_to_none() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .reasoning(ReasoningConfig {
                enabled: false,
                budget_tokens: None,
                effort: None,
            })
            .message(anyllm::Message::user("Hi"));

        let api_req = try_from_request(&req).unwrap();
        assert!(api_req.thinking.is_none());
    }

    #[test]
    fn reasoning_enabled_without_budget_uses_default() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: None,
                effort: None,
            })
            .message(anyllm::Message::user("Hi"));

        let api_req = try_from_request(&req).unwrap();
        let thinking = api_req.thinking.unwrap();
        assert_eq!(thinking.budget_tokens, DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn rejects_reasoning_budget_that_cannot_be_bumped() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514")
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: Some(u32::MAX),
                effort: None,
            })
            .message(anyllm::Message::user("Hi"));

        let err = try_from_request(&req).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::InvalidRequest(message) if message.contains("cannot be u32::MAX"))
        );
    }

    #[test]
    fn converts_multimodal_user_text_parts() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514").message(
            anyllm::Message::user_multimodal(vec![
                ContentPart::text("Describe this"),
                ContentPart::text("And this"),
            ]),
        );

        let api_req = try_from_request(&req).unwrap();
        assert_eq!(api_req.messages[0].content.len(), 2);
        match &api_req.messages[0].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Describe this"),
            other => panic!("expected Text, got {other:?}"),
        }
        match &api_req.messages[0].content[1] {
            ContentBlock::Text { text } => assert_eq!(text, "And this"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn converts_multimodal_user_image_only_message() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514").message(
            anyllm::Message::user_multimodal(vec![ContentPart::Image {
                source: anyllm::ImageSource::Url {
                    url: "https://example.com/cat.png".to_string(),
                },
                detail: None,
            }]),
        );

        let api_req = try_from_request(&req).unwrap();
        assert_eq!(api_req.messages.len(), 1);
        assert_eq!(api_req.messages[0].content.len(), 1);
        match &api_req.messages[0].content[0] {
            ContentBlock::Image { source } => match source {
                super::ImageSource::Url { url } => {
                    assert_eq!(url, "https://example.com/cat.png")
                }
                other => panic!("expected URL image source, got {other:?}"),
            },
            other => panic!("expected Image, got {other:?}"),
        }
    }

    #[test]
    fn converts_multimodal_user_mixed_text_and_base64_image() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514").message(
            anyllm::Message::user_multimodal(vec![
                ContentPart::text("Describe this image"),
                ContentPart::Image {
                    source: anyllm::ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: "iVBORw0KGgoAAAANSUhEUg==".to_string(),
                    },
                    detail: None,
                },
            ]),
        );

        let api_req = try_from_request(&req).unwrap();
        assert_eq!(api_req.messages[0].content.len(), 2);

        match &api_req.messages[0].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Describe this image"),
            other => panic!("expected Text, got {other:?}"),
        }
        match &api_req.messages[0].content[1] {
            ContentBlock::Image { source } => match source {
                super::ImageSource::Base64 { media_type, data } => {
                    assert_eq!(media_type, "image/png");
                    assert_eq!(data, "iVBORw0KGgoAAAANSUhEUg==");
                }
                other => panic!("expected base64 image source, got {other:?}"),
            },
            other => panic!("expected Image, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unsupported_user_part_type() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514").message(
            anyllm::Message::user_multimodal(vec![ContentPart::Other {
                type_name: "audio".to_string(),
                data: serde_json::Map::new(),
            }]),
        );

        let err = try_from_request(&req).unwrap_err();
        assert!(matches!(err, anyllm::Error::Unsupported(_)));
    }

    #[test]
    fn converts_text_response() {
        let api_resp = CreateMessageResponse {
            id: "msg_01234".to_string(),
            content: vec![ContentBlock::Text {
                text: "Hello!".to_string(),
            }],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some("end_turn".to_string()),
            usage: Usage {
                input_tokens: 100,
                output_tokens: 50,
                cache_read_input_tokens: Some(80),
                cache_creation_input_tokens: Some(10),
            },
        };

        let resp = try_into_response(api_resp).unwrap();

        assert_eq!(resp.text(), Some("Hello!".into()));
        assert_eq!(resp.finish_reason, Some(anyllm::FinishReason::Stop));
        assert_eq!(resp.model, Some("claude-sonnet-4-20250514".into()));
        assert_eq!(resp.id, Some("msg_01234".into()));

        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(50));
        assert_eq!(usage.cached_input_tokens, Some(80));
        assert_eq!(usage.cache_creation_input_tokens, Some(10));
    }

    #[test]
    fn converts_tool_use_response() {
        let api_resp = CreateMessageResponse {
            id: "msg_56789".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Let me check.".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_xyz".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path": "/etc/hosts"}),
                },
            ],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some("tool_use".to_string()),
            usage: Usage {
                input_tokens: 50,
                output_tokens: 30,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
        };

        let resp = try_into_response(api_resp).unwrap();

        assert_eq!(resp.finish_reason, Some(anyllm::FinishReason::ToolCalls));
        assert!(resp.has_tool_calls());
        let calls: Vec<_> = resp.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "toolu_xyz");
        assert_eq!(calls[0].name, "read_file");
        let args: serde_json::Value = serde_json::from_str(calls[0].arguments).unwrap();
        assert_eq!(args["path"], "/etc/hosts");
    }

    #[test]
    fn converts_image_response() {
        let api_resp = CreateMessageResponse {
            id: "msg_img".to_string(),
            content: vec![ContentBlock::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png".to_string(),
                },
            }],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some("end_turn".to_string()),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
        };

        let resp = try_into_response(api_resp).unwrap();
        assert!(matches!(
            &resp.content[..],
            [anyllm::ContentBlock::Image {
                source: anyllm::ImageSource::Url { url }
            }] if url == "https://example.com/cat.png"
        ));
    }

    #[test]
    fn converts_thinking_response() {
        let api_resp = CreateMessageResponse {
            id: "msg_think".to_string(),
            content: vec![
                ContentBlock::Thinking {
                    thinking: "Let me reason...".to_string(),
                    signature: Some("sig_abc".to_string()),
                },
                ContentBlock::Text {
                    text: "My answer.".to_string(),
                },
            ],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some("end_turn".to_string()),
            usage: Usage {
                input_tokens: 20,
                output_tokens: 100,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
        };

        let resp = try_into_response(api_resp).unwrap();

        assert_eq!(resp.reasoning_text(), Some("Let me reason...".into()));
        assert_eq!(resp.text(), Some("My answer.".into()));
        match &resp.content[0] {
            anyllm::ContentBlock::Reasoning { signature, .. } => {
                assert_eq!(signature, &Some("sig_abc".to_string()));
            }
            other => panic!("expected Reasoning, got {other:?}"),
        }
    }

    #[test]
    fn converts_response_with_null_stop_reason() {
        let api_resp = CreateMessageResponse {
            id: "msg_01234".to_string(),
            content: vec![ContentBlock::Text {
                text: "Streaming...".to_string(),
            }],
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
        };

        let resp = try_into_response(api_resp).unwrap();
        assert!(resp.finish_reason.is_none());
    }

    #[test]
    fn parses_known_stop_reasons() {
        assert_eq!(parse_stop_reason("end_turn"), anyllm::FinishReason::Stop);
        assert_eq!(
            parse_stop_reason("max_tokens"),
            anyllm::FinishReason::Length
        );
        assert_eq!(
            parse_stop_reason("tool_use"),
            anyllm::FinishReason::ToolCalls
        );
    }

    #[test]
    fn parses_unknown_stop_reason_as_other() {
        assert_eq!(
            parse_stop_reason("something_else"),
            anyllm::FinishReason::Other("something_else".into())
        );
    }

    #[test]
    fn rejects_invalid_json_in_assistant_tool_call_arguments() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514").message(
            anyllm::Message::Assistant {
                content: vec![anyllm::ContentBlock::ToolCall {
                    id: "toolu_abc".into(),
                    name: "read_file".into(),
                    arguments: "not valid json at all".into(),
                }],
                name: None,
                extensions: None,
            },
        );

        let err = try_from_request(&req).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Serialization(_)),
            "expected Serialization error, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("read_file"),
            "error should mention the tool name, got: {msg}"
        );
    }

    #[test]
    fn rejects_assistant_image_replay() {
        let req = anyllm::ChatRequest::new("claude-sonnet-4-20250514").message(
            anyllm::Message::Assistant {
                content: vec![anyllm::ContentBlock::Image {
                    source: anyllm::ImageSource::Url {
                        url: "https://example.com/cat.png".into(),
                    },
                }],
                name: None,
                extensions: None,
            },
        );

        let err = try_from_request(&req).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Unsupported(message) if message == "anthropic does not support assistant image replay")
        );
    }

    #[test]
    fn rejects_unsupported_portable_request_controls() {
        let unsupported_requests = [
            (
                anyllm::ChatRequest::new("claude-sonnet-4-20250514").parallel_tool_calls(true),
                "anthropic does not support parallel_tool_calls controls",
            ),
            (
                anyllm::ChatRequest::new("claude-sonnet-4-20250514").seed(42),
                "anthropic does not support seed controls",
            ),
            (
                anyllm::ChatRequest::new("claude-sonnet-4-20250514").frequency_penalty(0.2),
                "anthropic does not support frequency_penalty controls",
            ),
            (
                anyllm::ChatRequest::new("claude-sonnet-4-20250514").presence_penalty(0.3),
                "anthropic does not support presence_penalty controls",
            ),
        ];

        for (request, message) in unsupported_requests {
            let err = try_from_request(&request).unwrap_err();
            assert!(matches!(err, anyllm::Error::Unsupported(text) if text == message));
        }
    }

    #[test]
    fn req_system_prompt_emitted_as_top_level_system_block() {
        use anyllm::SystemPrompt;

        let mut req = ChatRequest::new("claude-3-5-sonnet-20240620").user("hi");
        req.system.push(SystemPrompt::new("Preamble"));

        let api_req = try_from_request(&req).unwrap();
        let value = serde_json::to_value(&api_req).unwrap();
        assert_eq!(value["system"][0]["type"], "text");
        assert_eq!(value["system"][0]["text"], "Preamble");
    }

    #[test]
    fn req_system_multiple_prompts_preserve_order() {
        use anyllm::SystemPrompt;

        let mut req = ChatRequest::new("claude-3-5-sonnet-20240620").user("hi");
        req.system.push(SystemPrompt::new("A"));
        req.system.push(SystemPrompt::new("B"));

        let api_req = try_from_request(&req).unwrap();
        let value = serde_json::to_value(&api_req).unwrap();
        assert_eq!(value["system"][0]["text"], "A");
        assert_eq!(value["system"][1]["text"], "B");
    }

    #[test]
    fn req_system_cache_control_option_is_emitted() {
        use crate::CacheControl;
        use anyllm::SystemPrompt;

        let mut req = ChatRequest::new("claude-3-5-sonnet-20240620").user("hi");
        req.system
            .push(SystemPrompt::new("Cached").with_option(CacheControl::ephemeral()));

        let api_req = try_from_request(&req).unwrap();
        let value = serde_json::to_value(&api_req).unwrap();
        assert_eq!(value["system"][0]["text"], "Cached");
        assert_eq!(value["system"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn req_system_empty_content_is_filtered() {
        use anyllm::SystemPrompt;

        let mut req = ChatRequest::new("claude-3-5-sonnet-20240620").user("hi");
        req.system.push(SystemPrompt::new(""));

        let api_req = try_from_request(&req).unwrap();
        let value: serde_json::Value = serde_json::to_value(&api_req).unwrap();
        let system_len = value
            .get("system")
            .and_then(|s| s.as_array())
            .map_or(0, Vec::len);
        assert_eq!(
            system_len, 0,
            "empty-content prompt should be filtered, got {value:?}"
        );
    }

    #[test]
    fn req_system_unknown_option_is_silently_ignored() {
        use anyllm::SystemPrompt;

        #[derive(Clone)]
        struct Foreign;

        let mut req = ChatRequest::new("claude-3-5-sonnet-20240620").user("hi");
        req.system.push(SystemPrompt::new("X").with_option(Foreign));

        let api_req = try_from_request(&req).unwrap();
        let value = serde_json::to_value(&api_req).unwrap();
        assert_eq!(value["system"][0]["text"], "X");
        assert!(value["system"][0].get("cache_control").is_none());
    }
}
