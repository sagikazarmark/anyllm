use anyllm::UserContent;
use anyllm::{
    ChatRequest, ChatResponse, ContentBlock, ContentPart, ExtraMap, FinishReason, ImageSource,
    Message, ResponseMetadata, ToolChoice, ToolResultContent, Usage,
};
use serde::{Deserialize, Serialize};

use crate::RequestOptions;
use crate::options::reasoning_effort_from_config;

const ROLE_SYSTEM: &str = "system";
const ROLE_USER: &str = "user";
const ROLE_ASSISTANT: &str = "assistant";
const ROLE_TOOL: &str = "tool";
const TOOL_TYPE_FUNCTION: &str = "function";

#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ApiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExtraMap>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(flatten, skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

#[derive(Debug, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ApiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ApiToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct ApiTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ApiFunctionDef,
}

#[derive(Debug, Serialize)]
pub struct ApiFunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub model: String,
    #[serde(default)]
    pub usage: Option<ApiUsage>,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionChoice {
    #[allow(dead_code)]
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionMessage {
    #[allow(dead_code)]
    pub role: String,
    #[serde(default, deserialize_with = "deserialize_string_or_json")]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ApiToolCall>>,
    #[serde(default)]
    pub refusal: Option<String>,
}

/// Deserialize a message `content` field that may arrive either as a JSON
/// string (per OpenAI spec) or as a structured JSON value.
///
/// Workaround (Cloudflare Workers AI, undocumented): when `response_format`
/// is a strict `json_schema`, the `/ai/v1/chat/completions` endpoint returns
/// `content` as a parsed JSON object (e.g. `{"greeting": "hello"}`) instead
/// of the stringified form OpenAI's spec requires. This accepts both shapes
/// and re-serializes non-string values back to their JSON string form so
/// downstream consumers see a uniform `String`.
fn deserialize_string_or_json<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.map(|value| match value {
        serde_json::Value::String(s) => s,
        other => other.to_string(),
    }))
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    #[allow(dead_code)]
    pub id: String,
    #[serde(default)]
    pub model: Option<String>,
    pub choices: Vec<ChunkChoice>,
    #[serde(default)]
    pub usage: Option<ApiUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkChoice {
    #[allow(dead_code)]
    pub index: u32,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkToolCall {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<ChunkToolCallFunction>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkToolCallFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub error: ApiError,
}

#[derive(Debug, Deserialize)]
pub struct ApiError {
    pub message: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub error_type: String,
    #[serde(default)]
    pub code: Option<String>,
}

pub fn to_chat_completion_request(
    request: &ChatRequest,
    stream: bool,
    provider_options: &RequestOptions,
) -> anyllm::Result<ChatCompletionRequest> {
    // Emit req.system as one role: "system" wire message per prompt, in order,
    // before any req.messages entries. Empty-content prompts are skipped so
    // we never put empty system messages on the wire. Typed SystemOptions
    // entries are silently ignored — no OpenAI-specific options are defined
    // in V1.
    let mut messages = Vec::with_capacity(request.system.len() + request.messages.len());
    for prompt in &request.system {
        if prompt.content.is_empty() {
            continue;
        }
        messages.push(ApiMessage {
            role: ROLE_SYSTEM.to_string(),
            content: Some(serde_json::json!(prompt.content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }
    for message in &request.messages {
        messages.push(convert_message(message)?);
    }

    let tools = request.tools.as_ref().and_then(|tools| {
        if tools.is_empty() {
            None
        } else {
            let mut api_tools = Vec::with_capacity(tools.len());
            for tool in tools {
                let strict = tool
                    .extensions
                    .as_ref()
                    .and_then(|extensions| extensions.get("strict"))
                    .and_then(|value| value.as_bool());

                let mut parameters = tool.parameters.clone();
                if strict == Some(true) {
                    inject_strict_additional_properties(&mut parameters);
                }

                api_tools.push(ApiTool {
                    tool_type: TOOL_TYPE_FUNCTION.to_string(),
                    function: ApiFunctionDef {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters,
                        strict,
                    },
                });
            }
            Some(api_tools)
        }
    });

    let tool_choice = request
        .tool_choice
        .as_ref()
        .map(|tool_choice| match tool_choice {
            ToolChoice::Auto => Ok(serde_json::json!("auto")),
            ToolChoice::Required => Ok(serde_json::json!("required")),
            ToolChoice::Disabled => Ok(serde_json::json!("none")),
            ToolChoice::Specific { name } => {
                Ok(serde_json::json!({"type": "function", "function": {"name": name}}))
            }
            _ => Err(anyllm::Error::Unsupported(
                "unknown tool choice is not supported by the OpenAI-compatible converter".into(),
            )),
        })
        .transpose()?;

    let response_format = request
        .response_format
        .as_ref()
        .map(|response_format| match response_format {
            anyllm::ResponseFormat::Text => Ok(serde_json::json!({"type": "text"})),
            anyllm::ResponseFormat::Json => Ok(serde_json::json!({"type": "json_object"})),
            anyllm::ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            } => {
                let mut schema = schema.clone();
                if *strict == Some(true) {
                    inject_strict_additional_properties(&mut schema);
                }

                let mut json_schema = serde_json::Map::new();
                if let Some(name) = name {
                    json_schema.insert("name".to_string(), serde_json::json!(name));
                }
                json_schema.insert("schema".to_string(), schema);
                if let Some(strict) = strict {
                    json_schema.insert("strict".to_string(), serde_json::json!(strict));
                }
                Ok(serde_json::json!({"type": "json_schema", "json_schema": json_schema}))
            }
            _ => Err(anyllm::Error::Unsupported(
                "unknown response format is not supported by the OpenAI-compatible converter"
                    .into(),
            )),
        })
        .transpose()?;

    let reasoning_effort = if let Some(effort) = &provider_options.reasoning_effort {
        Some(effort.to_string())
    } else {
        request
            .reasoning
            .as_ref()
            .and_then(reasoning_effort_from_config)
            .map(str::to_owned)
    };

    let stop = request.stop.as_ref().and_then(|sequences| {
        if sequences.is_empty() {
            None
        } else {
            Some(sequences.clone())
        }
    });

    let stream_options = if stream {
        Some(StreamOptions {
            include_usage: true,
        })
    } else {
        None
    };

    Ok(ChatCompletionRequest {
        model: request.model.clone(),
        messages,
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_tokens,
        stop,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        tools,
        tool_choice,
        response_format,
        seed: provider_options.seed.or(request.seed),
        reasoning_effort,
        parallel_tool_calls: provider_options
            .parallel_tool_calls
            .or(request.parallel_tool_calls),
        user: provider_options.user.clone(),
        service_tier: provider_options.service_tier.clone(),
        store: provider_options.store,
        metadata: provider_options.metadata.clone(),
        stream,
        stream_options,
        extra: provider_options.extra.clone(),
    })
}

/// Recursively inject `additionalProperties: false` into every object schema node.
///
/// OpenAI's strict mode (on `response_format` JSON schemas and on strict tool function
/// parameters) requires every object schema to explicitly set `additionalProperties: false`.
/// Most schema generators (including `schemars`) do not emit this by default, so without
/// intervention users have to annotate their types with `#[serde(deny_unknown_fields)]`.
///
/// This walker preserves any existing `additionalProperties` value the caller supplied so
/// they retain an explicit escape hatch. Nodes that declare `type: "object"` or expose a
/// `properties` map are treated as objects.
fn inject_strict_additional_properties(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            let is_object = map.get("type").and_then(|value| value.as_str()) == Some("object")
                || map.contains_key("properties");
            if is_object && !map.contains_key("additionalProperties") {
                map.insert(
                    "additionalProperties".to_string(),
                    serde_json::Value::Bool(false),
                );
            }

            for value in map.values_mut() {
                inject_strict_additional_properties(value);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                inject_strict_additional_properties(value);
            }
        }
        _ => {}
    }
}

pub fn convert_message(message: &Message) -> anyllm::Result<ApiMessage> {
    match message {
        Message::System { content, .. } => Ok(ApiMessage {
            role: ROLE_SYSTEM.to_string(),
            content: Some(serde_json::json!(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }),
        Message::User { content, name, .. } => {
            let api_content = match content {
                UserContent::Text(text) => serde_json::json!(text),
                UserContent::Parts(parts) => {
                    let mut api_parts = Vec::with_capacity(parts.len());
                    for part in parts {
                        api_parts.push(convert_user_part(part)?);
                    }
                    serde_json::json!(api_parts)
                }
            };
            Ok(ApiMessage {
                role: ROLE_USER.to_string(),
                content: Some(api_content),
                name: name.clone(),
                tool_calls: None,
                tool_call_id: None,
            })
        }
        Message::Assistant { content, name, .. } => {
            let mut text_content = String::new();
            let mut api_tool_calls = Vec::new();

            for block in content {
                match block {
                    ContentBlock::Text { text } => text_content.push_str(text),
                    ContentBlock::Image { .. } => {
                        return Err(anyllm::Error::Unsupported(
                            "OpenAI-compatible providers do not support assistant image replay via this converter"
                                .into(),
                        ));
                    }
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        api_tool_calls.push(ApiToolCall {
                            id: id.clone(),
                            call_type: TOOL_TYPE_FUNCTION.to_string(),
                            function: ApiToolCallFunction {
                                name: name.clone(),
                                arguments: arguments.clone(),
                            },
                        });
                    }
                    ContentBlock::Reasoning { .. } | ContentBlock::Other { .. } | _ => {}
                }
            }

            Ok(ApiMessage {
                role: ROLE_ASSISTANT.to_string(),
                content: if text_content.is_empty() {
                    None
                } else {
                    Some(serde_json::Value::String(text_content))
                },
                name: name.clone(),
                tool_calls: if api_tool_calls.is_empty() {
                    None
                } else {
                    Some(api_tool_calls)
                },
                tool_call_id: None,
            })
        }
        Message::Tool {
            tool_call_id,
            content,
            ..
        } => Ok(ApiMessage {
            role: ROLE_TOOL.to_string(),
            content: Some(convert_tool_result_content(content)?),
            name: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.clone()),
        }),
        _ => Err(anyllm::Error::Unsupported(
            "unknown message variant is not supported by the OpenAI-compatible converter".into(),
        )),
    }
}

fn convert_user_part(part: &ContentPart) -> anyllm::Result<serde_json::Value> {
    match part {
        ContentPart::Text { text } => Ok(serde_json::json!({"type": "text", "text": text})),
        ContentPart::Image { source, detail } => {
            let url = match source {
                ImageSource::Url { url } => url.clone(),
                ImageSource::Base64 { media_type, data } => {
                    format!("data:{media_type};base64,{data}")
                }
                _ => {
                    return Err(anyllm::Error::Unsupported(
                        "unknown image source is not supported by the OpenAI-compatible converter"
                            .into(),
                    ));
                }
            };
            let mut image_url = serde_json::Map::new();
            image_url.insert("url".to_string(), serde_json::Value::String(url));
            if let Some(detail) = detail {
                image_url.insert(
                    "detail".to_string(),
                    serde_json::Value::String(detail.clone()),
                );
            }
            Ok(serde_json::json!({"type": "image_url", "image_url": image_url}))
        }
        ContentPart::Other { type_name, data } => {
            let mut object = serde_json::Map::with_capacity(data.len() + 1);
            object.insert(
                "type".to_string(),
                serde_json::Value::String(type_name.clone()),
            );
            for (key, value) in data {
                object.insert(key.clone(), value.clone());
            }
            Ok(serde_json::Value::Object(object))
        }
        _ => Err(anyllm::Error::Unsupported(
            "unsupported user content part for OpenAI-compatible request conversion".into(),
        )),
    }
}

fn convert_tool_result_content(content: &ToolResultContent) -> anyllm::Result<serde_json::Value> {
    match content {
        ToolResultContent::Text(text) => Ok(serde_json::json!(text)),
        ToolResultContent::Parts(_) => Err(anyllm::Error::Unsupported(
            "OpenAI-compatible providers do not support multimodal tool result content via this converter"
                .into(),
        )),
    }
}

pub fn parse_finish_reason(value: &str) -> FinishReason {
    match value {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_string()),
    }
}

pub fn from_api_usage(usage: &ApiUsage) -> Usage {
    let cached_input_tokens = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens);
    let reasoning_tokens = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|details| details.reasoning_tokens);

    let mut converted = Usage::new()
        .input_tokens(usage.prompt_tokens)
        .output_tokens(usage.completion_tokens)
        .total_tokens(usage.total_tokens);
    converted.cached_input_tokens = cached_input_tokens;
    converted.reasoning_tokens = reasoning_tokens;
    converted
}

pub fn from_api_response<M>(
    response: ChatCompletionResponse,
    metadata_hook: M,
) -> anyllm::Result<ChatResponse>
where
    M: FnOnce(&ChatCompletionResponse, &mut ResponseMetadata),
{
    let mut metadata = ResponseMetadata::new();
    metadata_hook(&response, &mut metadata);

    let ChatCompletionResponse {
        id,
        choices,
        model,
        usage,
        system_fingerprint: _,
    } = response;

    let choice = choices
        .into_iter()
        .next()
        .ok_or_else(|| anyllm::Error::Provider {
            status: None,
            message: "OpenAI-compatible provider returned no choices".to_string(),
            body: None,
            request_id: None,
        })?;

    let ChatCompletionChoice {
        index: _,
        message,
        finish_reason,
    } = choice;

    let ChatCompletionMessage {
        role: _,
        content: message_content,
        tool_calls,
        refusal,
    } = message;

    let mut content = Vec::new();

    if let Some(text) = message_content
        && !text.is_empty()
    {
        content.push(ContentBlock::Text { text });
    }

    if let Some(refusal) = refusal
        && !refusal.is_empty()
    {
        content.push(ContentBlock::Text { text: refusal });
    }

    if let Some(tool_calls) = tool_calls {
        for tool_call in tool_calls {
            content.push(ContentBlock::ToolCall {
                id: tool_call.id,
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            });
        }
    }

    let mut converted = ChatResponse::new(content).metadata(metadata);
    converted.finish_reason = finish_reason.as_deref().map(parse_finish_reason);
    converted.usage = usage.as_ref().map(from_api_usage);
    converted.model = Some(model);
    converted.id = Some(id);
    Ok(converted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn inject_strict_adds_additional_properties_to_simple_object() {
        let mut schema = json!({
            "type": "object",
            "properties": {"greeting": {"type": "string"}},
            "required": ["greeting"]
        });
        inject_strict_additional_properties(&mut schema);
        assert_eq!(schema["additionalProperties"], json!(false));
    }

    #[test]
    fn inject_strict_recurses_into_nested_objects() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        });
        inject_strict_additional_properties(&mut schema);
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(
            schema["properties"]["user"]["additionalProperties"],
            json!(false)
        );
    }

    #[test]
    fn inject_strict_handles_array_items() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}}
                    }
                }
            }
        });
        inject_strict_additional_properties(&mut schema);
        assert_eq!(
            schema["properties"]["items"]["items"]["additionalProperties"],
            json!(false)
        );
    }

    #[test]
    fn inject_strict_handles_anyof_oneof_allof() {
        let mut schema = json!({
            "anyOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "string"}}}
            ]
        });
        inject_strict_additional_properties(&mut schema);
        assert_eq!(schema["anyOf"][0]["additionalProperties"], json!(false));
        assert_eq!(schema["anyOf"][1]["additionalProperties"], json!(false));
    }

    #[test]
    fn inject_strict_preserves_explicit_additional_properties() {
        let mut schema = json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": {"type": "integer"}
        });
        inject_strict_additional_properties(&mut schema);
        assert_eq!(schema["additionalProperties"], json!({"type": "integer"}));
    }

    #[test]
    fn inject_strict_detects_object_by_properties_key() {
        // Schema omits "type" but has "properties" — treat as object.
        let mut schema = json!({
            "properties": {"a": {"type": "string"}}
        });
        inject_strict_additional_properties(&mut schema);
        assert_eq!(schema["additionalProperties"], json!(false));
    }

    #[test]
    fn inject_strict_leaves_non_object_nodes_alone() {
        let mut schema = json!({
            "type": "string"
        });
        inject_strict_additional_properties(&mut schema);
        assert!(schema.get("additionalProperties").is_none());
    }

    #[test]
    fn chat_completion_message_accepts_string_content() {
        let raw = json!({
            "role": "assistant",
            "content": "hello world"
        });
        let message: ChatCompletionMessage = serde_json::from_value(raw).unwrap();
        assert_eq!(message.content.as_deref(), Some("hello world"));
    }

    #[test]
    fn chat_completion_message_coerces_object_content_to_json_string() {
        // Cloudflare Workers AI returns strict-schema responses with `content`
        // as a parsed JSON object instead of a stringified payload.
        let raw = json!({
            "role": "assistant",
            "content": {"greeting": "hello"}
        });
        let message: ChatCompletionMessage = serde_json::from_value(raw).unwrap();
        assert_eq!(message.content.as_deref(), Some(r#"{"greeting":"hello"}"#));
    }

    #[test]
    fn chat_completion_message_handles_missing_and_null_content() {
        let missing: ChatCompletionMessage =
            serde_json::from_value(json!({"role": "assistant"})).unwrap();
        assert!(missing.content.is_none());

        let null_content: ChatCompletionMessage =
            serde_json::from_value(json!({"role": "assistant", "content": null})).unwrap();
        assert!(null_content.content.is_none());
    }

    #[test]
    fn req_system_emits_system_messages_at_front() {
        use anyllm::SystemPrompt;

        let mut req = ChatRequest::new("gpt-4o-compat").user("hi");
        req.system.push(SystemPrompt::new("A"));
        req.system.push(SystemPrompt::new("B"));

        let api_req =
            to_chat_completion_request(&req, false, &RequestOptions::default()).unwrap();
        assert_eq!(api_req.messages[0].role, "system");
        assert_eq!(api_req.messages[0].content, Some(json!("A")));
        assert_eq!(api_req.messages[1].role, "system");
        assert_eq!(api_req.messages[1].content, Some(json!("B")));
        assert_eq!(api_req.messages[2].role, "user");
    }

    #[test]
    fn req_system_empty_emits_no_system_messages() {
        let req = ChatRequest::new("gpt-4o-compat").user("hi");
        let api_req =
            to_chat_completion_request(&req, false, &RequestOptions::default()).unwrap();
        assert_eq!(api_req.messages[0].role, "user");
    }

    #[test]
    fn req_system_empty_content_is_filtered() {
        use anyllm::SystemPrompt;
        let mut req = ChatRequest::new("gpt-4o-compat").user("hi");
        req.system.push(SystemPrompt::new(""));

        let api_req =
            to_chat_completion_request(&req, false, &RequestOptions::default()).unwrap();
        // Empty-content prompt should NOT produce a system wire message
        assert_eq!(api_req.messages[0].role, "user");
    }
}
