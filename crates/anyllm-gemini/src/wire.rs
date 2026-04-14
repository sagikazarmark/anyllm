use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub(crate) struct GenerateContentRequest {
    pub contents: Vec<Content>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(rename = "cachedContent", skip_serializing_if = "Option::is_none")]
    pub cached_content: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Content {
    pub role: String,
    pub parts: Vec<RequestPart>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SystemContent {
    pub parts: Vec<RequestPart>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum RequestPart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
    FileData {
        #[serde(rename = "fileData")]
        file_data: FileData,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCallPayload,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponsePayload,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FileData {
    pub mime_type: String,
    pub file_uri: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct FunctionCallPayload {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub(crate) struct FunctionResponsePayload {
    pub name: String,
    pub response: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub(crate) struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Serialize)]
pub(crate) struct FunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub(crate) struct ToolConfig {
    #[serde(rename = "functionCallingConfig")]
    pub function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Serialize)]
pub(crate) struct FunctionCallingConfig {
    pub mode: String,
    #[serde(
        rename = "allowedFunctionNames",
        skip_serializing_if = "Option::is_none"
    )]
    pub allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub(crate) struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(rename = "candidateCount", skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<u32>,
    #[serde(rename = "topK", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(rename = "responseMimeType", skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    #[serde(rename = "responseSchema", skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ThinkingConfig {
    #[serde(rename = "thinkingBudget")]
    pub thinking_budget: u32,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GenerateContentResponse {
    #[serde(rename = "responseId", default)]
    pub response_id: Option<String>,
    #[serde(rename = "modelVersion", default)]
    pub model_version: Option<String>,
    #[serde(rename = "promptFeedback", default)]
    pub prompt_feedback: Option<PromptFeedback>,
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata", default)]
    pub usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PromptFeedback {
    #[serde(rename = "blockReason", default)]
    pub block_reason: Option<String>,
    #[serde(rename = "blockReasonMessage", default)]
    pub block_reason_message: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Candidate {
    #[serde(default)]
    pub content: Option<ResponseContent>,
    #[serde(rename = "finishReason", default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct ResponseContent {
    #[serde(default)]
    pub parts: Vec<ResponsePart>,
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct ResponsePart {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(rename = "functionCall", default)]
    pub function_call: Option<FunctionCallPayload>,
    #[serde(rename = "inlineData", default)]
    pub inline_data: Option<InlineData>,
    #[serde(rename = "fileData", default)]
    pub file_data: Option<FileData>,
    #[serde(default)]
    pub thought: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct UsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    pub prompt_token_count: Option<u64>,
    #[serde(rename = "candidatesTokenCount", default)]
    pub candidates_token_count: Option<u64>,
    #[serde(rename = "totalTokenCount", default)]
    pub total_token_count: Option<u64>,
    #[serde(rename = "cachedContentTokenCount", default)]
    pub cached_content_token_count: Option<u64>,
    #[serde(rename = "thoughtsTokenCount", default)]
    pub thoughts_token_count: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiErrorResponse {
    pub error: GeminiError,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiError {
    #[serde(default)]
    #[allow(dead_code)]
    pub code: Option<u16>,
    pub message: String,
    #[serde(default)]
    pub status: Option<String>,
}

use anyllm::UserContent;
use anyllm::{
    ChatRequest, ChatResponse, ContentBlock, ContentPart, FinishReason, ImageSource, Message,
    ReasoningEffort, Result, ToolChoice, ToolResultContent, Usage,
};
use serde_json::json;
use uuid::Uuid;

use crate::ChatRequestOptions;

/// Convert an anyllm `ChatRequest` to a Gemini `GenerateContentRequest`.
pub(crate) fn to_api_request(request: &ChatRequest) -> Result<GenerateContentRequest> {
    let provider_options = request.option::<ChatRequestOptions>();
    let all_messages = &request.messages;

    // Extract system messages (all concatenated into one system instruction).
    let system_instruction = join_system_messages(all_messages).map(|text| SystemContent {
        parts: vec![RequestPart::Text { text }],
    });

    // Convert non-system messages to Gemini Content objects.
    let mut raw_contents: Vec<Content> = Vec::with_capacity(all_messages.len());
    let mut index = 0;
    while index < all_messages.len() {
        if matches!(all_messages[index], Message::Tool { .. }) {
            let run_start = index;
            while index < all_messages.len() && matches!(all_messages[index], Message::Tool { .. })
            {
                index += 1;
            }

            let mut tool_results: Vec<&Message> = all_messages[run_start..index].iter().collect();
            sort_consecutive_tool_results(&mut tool_results);
            for msg in tool_results {
                push_message_content(&mut raw_contents, msg)?;
            }
            continue;
        }

        push_message_content(&mut raw_contents, &all_messages[index])?;
        index += 1;
    }

    // Gemini requires strictly alternating user/model turns.
    // Merge consecutive same-role messages by concatenating their parts.
    let contents = enforce_alternating_turns(raw_contents);

    // Map tools.
    let tools = request.tools.as_ref().and_then(|tools| {
        if tools.is_empty() {
            None
        } else {
            Some(vec![GeminiTool {
                function_declarations: tools
                    .iter()
                    .map(|t| {
                        let mut parameters = t.parameters.clone();
                        sanitize_schema(&mut parameters);
                        FunctionDeclaration {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            parameters,
                        }
                    })
                    .collect(),
            }])
        }
    });

    // Map tool choice.
    let tool_config = request
        .tool_choice
        .as_ref()
        .map(|tc| {
            let (mode, allowed) = match tc {
                ToolChoice::Auto => Ok(("AUTO".to_string(), None)),
                ToolChoice::Disabled => Ok(("NONE".to_string(), None)),
                ToolChoice::Required => Ok(("ANY".to_string(), None)),
                ToolChoice::Specific { name } => Ok(("ANY".to_string(), Some(vec![name.clone()]))),
                _ => Err(anyllm::Error::Unsupported(
                    "unknown tool choice is not supported by the Gemini converter".into(),
                )),
            }?;
            Ok::<ToolConfig, anyllm::Error>(ToolConfig {
                function_calling_config: FunctionCallingConfig {
                    mode,
                    allowed_function_names: allowed,
                },
            })
        })
        .transpose()?;

    // Map generation config.
    let generation_config = build_generation_config(request)?;

    Ok(GenerateContentRequest {
        contents,
        system_instruction,
        tools,
        tool_config,
        generation_config,
        cached_content: provider_options.and_then(|opts| opts.cached_content.clone()),
    })
}

fn push_message_content(raw_contents: &mut Vec<Content>, msg: &Message) -> Result<()> {
    match msg {
        Message::System { .. } => {
            // Handled above as system_instruction.
        }
        Message::User { content, .. } => {
            let parts = convert_user_content(content)?;
            raw_contents.push(Content {
                role: "user".to_string(),
                parts,
            });
        }
        Message::Assistant { content, .. } => {
            let parts = convert_assistant_content(content)?;
            // An assistant message with no serializable parts is skipped.
            // This can happen when the assistant message contains only Reasoning blocks.
            if !parts.is_empty() {
                raw_contents.push(Content {
                    role: "model".to_string(),
                    parts,
                });
            }
        }
        Message::Tool {
            name,
            content,
            is_error,
            ..
        } => {
            let text = match content {
                ToolResultContent::Text(text) => text,
                ToolResultContent::Parts(_) => {
                    return Err(anyllm::Error::Unsupported(
                        "gemini converter does not support multimodal tool result content".into(),
                    ));
                }
            };
            // Wrap string content as a JSON object for Gemini's functionResponse.
            let response_val = if is_error.unwrap_or(false) {
                json!({"error": text})
            } else {
                // Try parsing as JSON object first; otherwise wrap as {output: ...}.
                serde_json::from_str::<serde_json::Value>(text)
                    .ok()
                    .filter(|v| v.is_object())
                    .unwrap_or_else(|| json!({"output": text}))
            };

            raw_contents.push(Content {
                role: "user".to_string(),
                parts: vec![RequestPart::FunctionResponse {
                    function_response: FunctionResponsePayload {
                        name: name.clone(),
                        response: response_val,
                    },
                }],
            });
        }
        _ => {
            return Err(anyllm::Error::Unsupported(
                "unknown message variant is not supported by the Gemini converter".into(),
            ));
        }
    }

    Ok(())
}

fn sort_consecutive_tool_results(tool_results: &mut Vec<&Message>) {
    let mut keyed = Vec::with_capacity(tool_results.len());
    for (original_index, msg) in tool_results.iter().enumerate() {
        let Message::Tool { tool_call_id, .. } = msg else {
            return;
        };

        let Some((scope, call_index)) = parse_gemini_tool_call_index(tool_call_id) else {
            return;
        };

        keyed.push((original_index, scope.to_string(), call_index));
    }

    if keyed.is_empty() {
        return;
    }

    let scope = &keyed[0].1;
    if keyed
        .iter()
        .any(|(_, candidate_scope, _)| candidate_scope != scope)
    {
        return;
    }

    keyed.sort_by_key(|(_, _, call_index)| *call_index);
    let sorted: Vec<&Message> = keyed
        .into_iter()
        .map(|(original_index, _, _)| tool_results[original_index])
        .collect();
    *tool_results = sorted;
}

pub(crate) fn parse_gemini_tool_call_index(tool_call_id: &str) -> Option<(&str, usize)> {
    let (scope, call_index) = tool_call_id.rsplit_once(":call:")?;
    let call_index = call_index.parse().ok()?;
    Some((scope, call_index))
}

/// Convert a `UserContent` to a list of Gemini request parts.
fn convert_user_content(content: &UserContent) -> Result<Vec<RequestPart>> {
    match content {
        UserContent::Text(text) => Ok(vec![RequestPart::Text { text: text.clone() }]),
        UserContent::Parts(parts) => {
            let mut converted = Vec::with_capacity(parts.len());
            for part in parts {
                let request_part = match part {
                    ContentPart::Text { text } => RequestPart::Text { text: text.clone() },
                    ContentPart::Image { source, .. } => request_part_from_image_source(source)?,
                    ContentPart::Other { type_name, data } => {
                        let _ = data;
                        return Err(anyllm::Error::Unsupported(format!(
                            "ContentPart::Other (type '{type_name}') is not supported by the Gemini provider"
                        )));
                    }
                    _ => {
                        return Err(anyllm::Error::Unsupported(
                            "unsupported user content part for Gemini conversion".into(),
                        ));
                    }
                };
                converted.push(request_part);
            }
            Ok(converted)
        }
    }
}

/// Convert assistant `ContentBlock`s to Gemini request parts.
fn convert_assistant_content(content: &[ContentBlock]) -> Result<Vec<RequestPart>> {
    let mut parts: Vec<RequestPart> = Vec::with_capacity(content.len());
    for block in content {
        match block {
            ContentBlock::Text { text } => {
                parts.push(RequestPart::Text { text: text.clone() });
            }
            ContentBlock::Image { source } => {
                parts.push(request_part_from_image_source(source)?);
            }
            ContentBlock::ToolCall {
                name, arguments, ..
            } => {
                // Gemini's functionCall.args is a JSON Value (not a JSON string).
                let args: serde_json::Value =
                    serde_json::from_str(arguments).map_err(anyllm::Error::from)?;
                parts.push(RequestPart::FunctionCall {
                    function_call: FunctionCallPayload {
                        name: name.clone(),
                        args,
                    },
                });
            }
            // Reasoning blocks cannot be round-tripped through Gemini (no thought input).
            ContentBlock::Reasoning { .. } | ContentBlock::Other { .. } | _ => {}
        }
    }
    Ok(parts)
}

fn join_system_messages(messages: &[Message]) -> Option<String> {
    let mut joined = String::new();

    for message in messages {
        let Message::System { content, .. } = message else {
            continue;
        };

        if !joined.is_empty() {
            joined.push_str("\n\n");
        }
        joined.push_str(content);
    }

    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

fn request_part_from_image_source(source: &ImageSource) -> Result<RequestPart> {
    match source {
        ImageSource::Base64 { media_type, data } => Ok(RequestPart::InlineData {
            inline_data: InlineData {
                mime_type: media_type.clone(),
                data: data.clone(),
            },
        }),
        ImageSource::Url { url } => {
            let mime_type = infer_mime_type(url).ok_or_else(|| {
                anyllm::Error::Unsupported(format!(
                    "Gemini cannot infer a supported MIME type from image URL '{url}'"
                ))
            })?;
            Ok(RequestPart::FileData {
                file_data: FileData {
                    mime_type,
                    file_uri: url.clone(),
                },
            })
        }
        _ => Err(anyllm::Error::Unsupported(
            "unknown image source is not supported by the Gemini converter".into(),
        )),
    }
}

/// Merge consecutive same-role `Content` entries by appending their parts.
///
/// Gemini requires strictly alternating user/model turns.
/// Gemini accepts a subset of OpenAPI 3.0 schema and rejects JSON Schema
/// meta-fields like `$schema` and `additionalProperties`. Strip them
/// recursively so schemas produced by `schemars` (via the extract feature)
/// or by callers using JSON Schema idioms still round-trip cleanly.
fn sanitize_schema(value: &mut serde_json::Value) {
    const UNSUPPORTED_KEYS: &[&str] = &[
        "$schema",
        "$id",
        "$defs",
        "definitions",
        "additionalProperties",
    ];

    match value {
        serde_json::Value::Object(map) => {
            for key in UNSUPPORTED_KEYS {
                map.remove(*key);
            }
            for (_, child) in map.iter_mut() {
                sanitize_schema(child);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                sanitize_schema(item);
            }
        }
        _ => {}
    }
}

fn enforce_alternating_turns(contents: Vec<Content>) -> Vec<Content> {
    let mut result: Vec<Content> = Vec::new();
    for content in contents {
        match result.last_mut() {
            Some(last) if last.role == content.role => {
                last.parts.extend(content.parts);
            }
            _ => result.push(content),
        }
    }
    result
}

/// Build a `GenerationConfig` from the request fields. Returns `None` if all fields are absent.
fn build_generation_config(request: &ChatRequest) -> Result<Option<GenerationConfig>> {
    let provider_options = request.option::<ChatRequestOptions>();
    let stop_sequences = request.stop.as_ref().and_then(|seqs| {
        if seqs.is_empty() {
            None
        } else {
            Some(seqs.clone())
        }
    });

    let (response_mime_type, response_schema) = match &request.response_format {
        Some(anyllm::ResponseFormat::Json) => (Some("application/json".to_string()), None),
        Some(anyllm::ResponseFormat::JsonSchema { schema, .. }) => {
            let mut schema = schema.clone();
            sanitize_schema(&mut schema);
            (Some("application/json".to_string()), Some(schema))
        }
        Some(anyllm::ResponseFormat::Text) | None | Some(_) => (None, None),
    };

    // Provider-specific thinking_budget overrides ChatRequest.reasoning.
    let provider_thinking_budget = provider_options.and_then(|opts| opts.thinking_budget);
    let thinking_config = if let Some(budget) = provider_thinking_budget {
        Some(ThinkingConfig {
            thinking_budget: budget,
        })
    } else {
        request.reasoning.as_ref().and_then(|rc| {
            if !rc.enabled {
                return None;
            }
            let budget = rc.budget_tokens.unwrap_or({
                if matches!(rc.effort, Some(ReasoningEffort::Low)) {
                    1024
                } else if matches!(rc.effort, Some(ReasoningEffort::Medium) | None) {
                    8192
                } else {
                    24576
                }
            });
            Some(ThinkingConfig {
                thinking_budget: budget,
            })
        })
    };

    let provider_response_mime_type =
        provider_options.and_then(|opts| opts.response_mime_type.clone());
    let provider_candidate_count = provider_options.and_then(|opts| opts.candidate_count);
    let provider_candidate_count = match provider_candidate_count {
        Some(0) => {
            return Err(anyllm::Error::InvalidRequest(
                "gemini candidate_count must be at least 1".into(),
            ));
        }
        Some(1) | None => provider_candidate_count,
        Some(_) => {
            return Err(anyllm::Error::Unsupported(
                "gemini candidate_count > 1 is not yet supported by anyllm".into(),
            ));
        }
    };
    let provider_top_k = provider_options.and_then(|opts| opts.top_k);

    let has_any = request.temperature.is_some()
        || request.top_p.is_some()
        || request.max_tokens.is_some()
        || provider_candidate_count.is_some()
        || provider_top_k.is_some()
        || stop_sequences.is_some()
        || response_mime_type.is_some()
        || provider_response_mime_type.is_some()
        || thinking_config.is_some();

    if !has_any {
        return Ok(None);
    }

    Ok(Some(GenerationConfig {
        temperature: request.temperature,
        candidate_count: provider_candidate_count,
        top_k: provider_top_k,
        top_p: request.top_p,
        max_output_tokens: request.max_tokens,
        stop_sequences,
        response_mime_type: provider_response_mime_type.or(response_mime_type),
        response_schema,
        thinking_config,
    }))
}

/// Infer a MIME type from a URL path's file extension (best-effort).
fn infer_mime_type(url: &str) -> Option<String> {
    let path = url
        .split(['?', '#'])
        .next()
        .unwrap_or(url)
        .rsplit('/')
        .next()
        .unwrap_or(url)
        .to_ascii_lowercase();

    if path.ends_with(".png") {
        Some("image/png".to_string())
    } else if path.ends_with(".jpg") || path.ends_with(".jpeg") {
        Some("image/jpeg".to_string())
    } else if path.ends_with(".gif") {
        Some("image/gif".to_string())
    } else if path.ends_with(".webp") {
        Some("image/webp".to_string())
    } else {
        None
    }
}

/// Convert a Gemini `GenerateContentResponse` to an anyllm `ChatResponse`.
pub(crate) fn from_api_response(response: GenerateContentResponse) -> Result<ChatResponse> {
    let GenerateContentResponse {
        response_id,
        model_version,
        prompt_feedback,
        candidates,
        usage_metadata,
    } = response;

    if let Some(err) = prompt_feedback.as_ref().and_then(prompt_feedback_error) {
        return Err(err);
    }

    let usage = usage_metadata.map(from_usage_metadata);

    let (blocks, finish_reason) = match candidates.into_iter().next() {
        Some(candidate) => {
            let fallback_tool_call_scope =
                response_id.is_none().then(|| Uuid::new_v4().to_string());
            let content = candidate_to_content_blocks(
                candidate,
                response_id.as_deref(),
                fallback_tool_call_scope.as_deref(),
            )?;
            (content.blocks, content.finish_reason)
        }
        None => (Vec::new(), None),
    };

    let mut response = ChatResponse::new(blocks);
    response.finish_reason = finish_reason;
    response.usage = usage;
    response.model = model_version;
    response.id = response_id;
    Ok(response)
}

pub(crate) fn prompt_feedback_error(prompt_feedback: &PromptFeedback) -> Option<anyllm::Error> {
    let block_reason = prompt_feedback.block_reason.as_deref()?;
    let message = prompt_feedback
        .block_reason_message
        .clone()
        .unwrap_or_else(|| format!("Prompt was blocked by Gemini: {block_reason}"));

    Some(match block_reason {
        "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "IMAGE_SAFETY" => {
            anyllm::Error::ContentFiltered(message)
        }
        _ => anyllm::Error::InvalidRequest(message),
    })
}

#[cfg(test)]
pub(crate) fn conformance_to_api_request(request: &ChatRequest) -> Result<serde_json::Value> {
    serde_json::to_value(to_api_request(request)?).map_err(anyllm::Error::from)
}

#[cfg(test)]
pub(crate) fn conformance_from_api_response_value(
    value: serde_json::Value,
) -> Result<ChatResponse> {
    let response: GenerateContentResponse =
        serde_json::from_value(value).map_err(anyllm::Error::from)?;
    from_api_response(response)
}

struct CandidateContent {
    blocks: Vec<ContentBlock>,
    finish_reason: Option<FinishReason>,
}

pub(crate) fn synthetic_tool_call_id(
    response_id: Option<&str>,
    fallback_scope: Option<&str>,
    call_index: usize,
) -> String {
    match response_id {
        Some(response_id) => format!("gemini:{response_id}:call:{call_index}"),
        None => format!(
            "gemini:anonymous:{}:call:{call_index}",
            fallback_scope
                .unwrap_or_else(|| unreachable!("fallback scope required without response id"))
        ),
    }
}

fn candidate_to_content_blocks(
    candidate: Candidate,
    response_id: Option<&str>,
    fallback_tool_call_scope: Option<&str>,
) -> Result<CandidateContent> {
    let raw_finish_reason = candidate.finish_reason.as_deref().map(parse_finish_reason);

    let mut blocks: Vec<ContentBlock> = Vec::new();
    let mut fc_index: usize = 0;

    if let Some(content) = candidate.content {
        for part in content.parts {
            if let Some(fc) = part.function_call {
                // Gemini args are a JSON object; convert to JSON string for ContentBlock.
                let arguments = serde_json::to_string(&fc.args).map_err(anyllm::Error::from)?;
                blocks.push(ContentBlock::ToolCall {
                    id: synthetic_tool_call_id(response_id, fallback_tool_call_scope, fc_index),
                    name: fc.name,
                    arguments,
                });
                fc_index += 1;
            } else if let Some(inline_data) = part.inline_data {
                blocks.push(ContentBlock::Image {
                    source: ImageSource::Base64 {
                        media_type: inline_data.mime_type,
                        data: inline_data.data,
                    },
                });
            } else if let Some(file_data) = part.file_data {
                blocks.push(ContentBlock::Image {
                    source: ImageSource::Url {
                        url: file_data.file_uri,
                    },
                });
            } else if let Some(text) = part.text
                && !text.is_empty()
            {
                if part.thought == Some(true) {
                    blocks.push(ContentBlock::Reasoning {
                        text,
                        signature: None,
                    });
                } else {
                    blocks.push(ContentBlock::Text { text });
                }
            }
        }
    }

    // Gemini returns "STOP" even when the response contains tool calls.
    // Normalize: if tool calls are present, finish_reason must be ToolCalls
    // so consumers can rely on it for routing.
    let has_tool_calls = blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::ToolCall { .. }));
    let finish_reason = if has_tool_calls {
        Some(FinishReason::ToolCalls)
    } else {
        raw_finish_reason
    };

    Ok(CandidateContent {
        blocks,
        finish_reason,
    })
}

/// Parse a Gemini `finishReason` string to an anyllm `FinishReason`.
pub(crate) fn parse_finish_reason(s: &str) -> FinishReason {
    match s {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" | "PROHIBITED_CONTENT" | "BLOCKLIST" | "IMAGE_SAFETY" => {
            FinishReason::ContentFilter
        }
        // MALFORMED_FUNCTION_CALL indicates the model tried to call a tool but failed.
        // Surface as ToolCalls (best effort) so callers can detect tool-related endings.
        "MALFORMED_FUNCTION_CALL" => FinishReason::ToolCalls,
        other => FinishReason::Other(other.to_string()),
    }
}

/// Convert Gemini `UsageMetadata` to anyllm `Usage`.
pub(crate) fn from_usage_metadata(meta: UsageMetadata) -> Usage {
    let mut usage = Usage::new();
    usage.input_tokens = meta.prompt_token_count;
    usage.output_tokens = meta.candidates_token_count;
    usage.total_tokens = meta.total_token_count;
    usage.cached_input_tokens = meta.cached_content_token_count;
    usage.reasoning_tokens = meta.thoughts_token_count;
    usage
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyllm::{ChatRequest, ContentBlock, Message, ReasoningConfig, ResponseFormat, Tool};
    use serde_json::json;

    #[test]
    fn converts_simple_user_message() {
        let req = ChatRequest::new("gemini-2.0-flash").message(Message::user("Hello!"));
        let api_req = to_api_request(&req).unwrap();

        assert_eq!(api_req.contents.len(), 1);
        assert_eq!(api_req.contents[0].role, "user");
        match &api_req.contents[0].parts[0] {
            RequestPart::Text { text } => assert_eq!(text, "Hello!"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert!(api_req.system_instruction.is_none());
    }

    #[test]
    fn extracts_system_messages_to_system_instruction() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .system("You are helpful.")
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req).unwrap();

        // System is extracted — contents only has the user message.
        assert_eq!(api_req.contents.len(), 1);
        assert_eq!(api_req.contents[0].role, "user");

        let si = api_req.system_instruction.unwrap();
        match &si.parts[0] {
            RequestPart::Text { text } => assert_eq!(text, "You are helpful."),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn concatenates_multiple_system_messages() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .message(Message::System {
                content: "First instruction.".to_string(),
                extensions: None,
            })
            .message(Message::System {
                content: "Second instruction.".to_string(),
                extensions: None,
            })
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req).unwrap();

        let si = api_req.system_instruction.unwrap();
        match &si.parts[0] {
            RequestPart::Text { text } => {
                assert!(text.contains("First instruction."));
                assert!(text.contains("Second instruction."));
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn merges_consecutive_user_messages() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .message(Message::user("First question."))
            .message(Message::user("Second question."));

        let api_req = to_api_request(&req).unwrap();

        // Two consecutive user messages → merged into one Content with two parts.
        assert_eq!(api_req.contents.len(), 1);
        assert_eq!(api_req.contents[0].role, "user");
        assert_eq!(api_req.contents[0].parts.len(), 2);
    }

    #[test]
    fn preserves_alternating_turns() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .message(Message::user("Hi"))
            .message(Message::Assistant {
                content: vec![ContentBlock::Text {
                    text: "Hello!".to_string(),
                }],
                name: None,
                extensions: None,
            })
            .message(Message::user("How are you?"));

        let api_req = to_api_request(&req).unwrap();

        assert_eq!(api_req.contents.len(), 3);
        assert_eq!(api_req.contents[0].role, "user");
        assert_eq!(api_req.contents[1].role, "model");
        assert_eq!(api_req.contents[2].role, "user");
    }

    #[test]
    fn converts_assistant_text() {
        let req = ChatRequest::new("gemini-2.0-flash").message(Message::Assistant {
            content: vec![ContentBlock::Text {
                text: "I can help!".to_string(),
            }],
            name: None,
            extensions: None,
        });

        let api_req = to_api_request(&req).unwrap();
        assert_eq!(api_req.contents[0].role, "model");
        match &api_req.contents[0].parts[0] {
            RequestPart::Text { text } => assert_eq!(text, "I can help!"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn converts_assistant_tool_call_to_function_call_part() {
        let req = ChatRequest::new("gemini-2.0-flash").message(Message::Assistant {
            content: vec![ContentBlock::ToolCall {
                id: "call_0".to_string(),
                name: "read_file".to_string(),
                arguments: r#"{"path": "/tmp/test.txt"}"#.to_string(),
            }],
            name: None,
            extensions: None,
        });

        let api_req = to_api_request(&req).unwrap();
        assert_eq!(api_req.contents[0].role, "model");
        match &api_req.contents[0].parts[0] {
            RequestPart::FunctionCall { function_call } => {
                assert_eq!(function_call.name, "read_file");
                assert_eq!(function_call.args["path"], "/tmp/test.txt");
            }
            other => panic!("expected FunctionCall, got {other:?}"),
        }
    }

    #[test]
    fn drops_reasoning_blocks_from_assistant_content() {
        let req = ChatRequest::new("gemini-2.0-flash").message(Message::Assistant {
            content: vec![
                ContentBlock::Reasoning {
                    text: "thinking...".to_string(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "Answer".to_string(),
                },
            ],
            name: None,
            extensions: None,
        });

        let api_req = to_api_request(&req).unwrap();
        assert_eq!(api_req.contents[0].parts.len(), 1);
        match &api_req.contents[0].parts[0] {
            RequestPart::Text { text } => assert_eq!(text, "Answer"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn assistant_with_only_reasoning_skipped() {
        // Assistant message with only Reasoning blocks → no Content emitted.
        let req = ChatRequest::new("gemini-2.0-flash")
            .message(Message::Assistant {
                content: vec![ContentBlock::Reasoning {
                    text: "Just thinking.".to_string(),
                    signature: None,
                }],
                name: None,
                extensions: None,
            })
            .message(Message::user("Reply"));

        let api_req = to_api_request(&req).unwrap();
        // Only the user message should be in contents.
        assert_eq!(api_req.contents.len(), 1);
        assert_eq!(api_req.contents[0].role, "user");
    }

    #[test]
    fn converts_tool_result_using_function_name_from_history() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .message(Message::user("What's in /tmp?"))
            .message(Message::Assistant {
                content: vec![ContentBlock::ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: r#"{"path": "/tmp"}"#.to_string(),
                }],
                name: None,
                extensions: None,
            })
            .message(Message::tool_result(
                "call_0",
                "read_file",
                "file1.txt\nfile2.txt",
            ));

        let api_req = to_api_request(&req).unwrap();

        // After alternating-turn enforcement: user → model → user (tool result merged)
        let last = api_req.contents.last().unwrap();
        assert_eq!(last.role, "user");
        match &last.parts[0] {
            RequestPart::FunctionResponse { function_response } => {
                assert_eq!(function_response.name, "read_file");
                assert_eq!(function_response.response["output"], "file1.txt\nfile2.txt");
            }
            other => panic!("expected FunctionResponse, got {other:?}"),
        }
    }

    #[test]
    fn error_tool_result_uses_error_key() {
        let req = ChatRequest::new("gemini-2.0-flash").message(Message::Tool {
            tool_call_id: "call_0".to_string(),
            name: "my_tool".to_string(),
            content: anyllm::ToolResultContent::text("Something went wrong"),
            is_error: Some(true),
            extensions: None,
        });

        let api_req = to_api_request(&req).unwrap();
        let last = api_req.contents.last().unwrap();
        match &last.parts[0] {
            RequestPart::FunctionResponse { function_response } => {
                assert_eq!(function_response.response["error"], "Something went wrong");
            }
            other => panic!("expected FunctionResponse, got {other:?}"),
        }
    }

    #[test]
    fn converts_tool_result_uses_matching_prior_assistant_when_ids_repeat_across_turns() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .message(Message::Assistant {
                content: vec![ContentBlock::ToolCall {
                    id: "call_0".to_string(),
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                }],
                name: None,
                extensions: None,
            })
            .message(Message::tool_result("call_0", "read_file", "first"))
            .message(Message::Assistant {
                content: vec![ContentBlock::ToolCall {
                    id: "call_0".to_string(),
                    name: "search".to_string(),
                    arguments: "{}".to_string(),
                }],
                name: None,
                extensions: None,
            })
            .message(Message::tool_result("call_0", "search", "second"));

        let api_req = to_api_request(&req).unwrap();
        let function_response_names: Vec<_> = api_req
            .contents
            .iter()
            .filter(|content| content.role == "user")
            .flat_map(|content| content.parts.iter())
            .filter_map(|part| match part {
                RequestPart::FunctionResponse { function_response } => {
                    Some(function_response.name.as_str())
                }
                _ => None,
            })
            .collect();

        assert_eq!(function_response_names, vec!["read_file", "search"]);
    }

    #[test]
    fn tool_result_uses_name_field_for_function_response() {
        let req = ChatRequest::new("gemini-2.0-flash").message(Message::tool_result(
            "call_0",
            "read_file",
            "contents",
        ));

        let api_req = to_api_request(&req).unwrap();
        let last = api_req.contents.last().unwrap();
        match &last.parts[0] {
            RequestPart::FunctionResponse { function_response } => {
                assert_eq!(function_response.name, "read_file");
            }
            other => panic!("expected FunctionResponse, got {other:?}"),
        }
    }

    #[test]
    fn consecutive_tool_results_are_sorted_by_gemini_call_index() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .message(Message::Assistant {
                content: vec![
                    ContentBlock::ToolCall {
                        id: "gemini:resp_123:call:0".to_string(),
                        name: "search".to_string(),
                        arguments: r#"{"query":"rust"}"#.to_string(),
                    },
                    ContentBlock::ToolCall {
                        id: "gemini:resp_123:call:1".to_string(),
                        name: "search".to_string(),
                        arguments: r#"{"query":"python"}"#.to_string(),
                    },
                ],
                name: None,
                extensions: None,
            })
            .message(Message::tool_result(
                "gemini:resp_123:call:1",
                "search",
                "python result",
            ))
            .message(Message::tool_result(
                "gemini:resp_123:call:0",
                "search",
                "rust result",
            ));

        let api_req = to_api_request(&req).unwrap();
        let last = api_req.contents.last().unwrap();
        let responses: Vec<_> = last
            .parts
            .iter()
            .filter_map(|part| match part {
                RequestPart::FunctionResponse { function_response } => {
                    Some(function_response.response["output"].as_str().unwrap())
                }
                _ => None,
            })
            .collect();

        assert_eq!(responses, vec!["rust result", "python result"]);
    }

    #[test]
    fn maps_tools_into_single_gemini_tool() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .tools(vec![
                Tool::new("search", json!({"type": "object"})).description("Search the web"),
                Tool::new("read_file", json!({"type": "object"})).description("Read a file"),
            ])
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req).unwrap();
        let tools = api_req.tools.unwrap();
        // All tools go into a single GeminiTool with multiple function_declarations.
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function_declarations.len(), 2);
        assert_eq!(tools[0].function_declarations[0].name, "search");
        assert_eq!(tools[0].function_declarations[1].name, "read_file");
    }

    #[test]
    fn empty_tools_vec_maps_to_none() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .tools(vec![])
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        assert!(api_req.tools.is_none());
    }

    #[test]
    fn tool_choice_auto() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .tools(vec![Tool::new("t", json!({})).description("d")])
            .tool_choice(ToolChoice::Auto)
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        let tc = api_req.tool_config.unwrap();
        assert_eq!(tc.function_calling_config.mode, "AUTO");
        assert!(tc.function_calling_config.allowed_function_names.is_none());
    }

    #[test]
    fn tool_choice_disabled_maps_to_none_mode() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .tools(vec![Tool::new("t", json!({})).description("d")])
            .tool_choice(ToolChoice::Disabled)
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        let tc = api_req.tool_config.unwrap();
        assert_eq!(tc.function_calling_config.mode, "NONE");
    }

    #[test]
    fn tool_choice_required_maps_to_any_mode() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .tools(vec![Tool::new("t", json!({})).description("d")])
            .tool_choice(ToolChoice::Required)
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        let tc = api_req.tool_config.unwrap();
        assert_eq!(tc.function_calling_config.mode, "ANY");
    }

    #[test]
    fn tool_choice_specific_maps_to_any_with_allowed_names() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .tools(vec![Tool::new("read_file", json!({})).description("d")])
            .tool_choice(ToolChoice::Specific {
                name: "read_file".into(),
            })
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        let tc = api_req.tool_config.unwrap();
        assert_eq!(tc.function_calling_config.mode, "ANY");
        assert_eq!(
            tc.function_calling_config.allowed_function_names,
            Some(vec!["read_file".to_string()])
        );
    }

    #[test]
    fn maps_temperature_and_top_p() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .temperature(0.7)
            .top_p(0.9)
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        let gc = api_req.generation_config.unwrap();
        assert_eq!(gc.temperature, Some(0.7));
        assert_eq!(gc.top_p, Some(0.9));
    }

    #[test]
    fn maps_max_tokens_to_max_output_tokens() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .max_tokens(1024)
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        let gc = api_req.generation_config.unwrap();
        assert_eq!(gc.max_output_tokens, Some(1024));
    }

    #[test]
    fn maps_response_format_json() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .response_format(ResponseFormat::Json)
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        let gc = api_req.generation_config.unwrap();
        assert_eq!(gc.response_mime_type, Some("application/json".to_string()));
        assert!(gc.response_schema.is_none());
    }

    #[test]
    fn maps_response_format_json_schema() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .response_format(ResponseFormat::JsonSchema {
                name: Some("person".into()),
                schema: json!({"type": "object", "properties": {"name": {"type": "string"}}}),
                strict: None,
            })
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        let gc = api_req.generation_config.unwrap();
        assert_eq!(gc.response_mime_type, Some("application/json".to_string()));
        assert!(gc.response_schema.is_some());
    }

    #[test]
    fn provider_options_override_generation_config_and_cached_content() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .response_format(ResponseFormat::Json)
            .with_option(ChatRequestOptions {
                top_k: Some(40),
                candidate_count: Some(1),
                response_mime_type: Some("text/x.enum".into()),
                cached_content: Some("cachedContents/abc123".into()),
                thinking_budget: None,
            })
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req).unwrap();
        let gc = api_req.generation_config.unwrap();

        assert_eq!(gc.top_k, Some(40));
        assert_eq!(gc.candidate_count, Some(1));
        assert_eq!(gc.response_mime_type, Some("text/x.enum".to_string()));
        assert_eq!(
            api_req.cached_content.as_deref(),
            Some("cachedContents/abc123")
        );
    }

    #[test]
    fn provider_only_generation_options_still_emit_generation_config() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .with_option(ChatRequestOptions {
                top_k: Some(40),
                candidate_count: Some(1),
                response_mime_type: Some("text/x.enum".into()),
                cached_content: None,
                thinking_budget: None,
            })
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req).unwrap();
        let gc = api_req
            .generation_config
            .expect("provider-only generation options should emit generation_config");

        assert_eq!(gc.top_k, Some(40));
        assert_eq!(gc.candidate_count, Some(1));
        assert_eq!(gc.response_mime_type, Some("text/x.enum".to_string()));
    }

    #[test]
    fn rejects_candidate_count_greater_than_one() {
        let req = ChatRequest::new("gemini-2.0-flash")
            .with_option(ChatRequestOptions {
                top_k: None,
                candidate_count: Some(2),
                response_mime_type: None,
                cached_content: None,
                thinking_budget: None,
            })
            .message(Message::user("Hi"));

        let err = to_api_request(&req).unwrap_err();
        assert!(matches!(
            err,
            anyllm::Error::Unsupported(message)
                if message.contains("candidate_count > 1")
        ));
    }

    #[test]
    fn maps_reasoning_budget_tokens() {
        let req = ChatRequest::new("gemini-2.5-pro")
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: Some(8192),
                effort: None,
            })
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        let gc = api_req.generation_config.unwrap();
        let tc = gc.thinking_config.unwrap();
        assert_eq!(tc.thinking_budget, 8192);
    }

    #[test]
    fn maps_reasoning_effort_low_to_budget_1024() {
        let req = ChatRequest::new("gemini-2.5-pro")
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: None,
                effort: Some(ReasoningEffort::Low),
            })
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        let tc = api_req.generation_config.unwrap().thinking_config.unwrap();
        assert_eq!(tc.thinking_budget, 1024);
    }

    #[test]
    fn maps_reasoning_effort_high_to_budget_24576() {
        let req = ChatRequest::new("gemini-2.5-pro")
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: None,
                effort: Some(ReasoningEffort::High),
            })
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        let tc = api_req.generation_config.unwrap().thinking_config.unwrap();
        assert_eq!(tc.thinking_budget, 24576);
    }

    #[test]
    fn reasoning_disabled_omits_thinking_config() {
        let req = ChatRequest::new("gemini-2.5-pro")
            .reasoning(ReasoningConfig {
                enabled: false,
                budget_tokens: Some(8192),
                effort: Some(ReasoningEffort::High),
            })
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        // No other gen config fields set, so generation_config itself is None.
        assert!(api_req.generation_config.is_none());
    }

    #[test]
    fn no_generation_config_when_all_none() {
        let req = ChatRequest::new("gemini-2.0-flash").message(Message::user("Hi"));
        let api_req = to_api_request(&req).unwrap();
        assert!(api_req.generation_config.is_none());
    }

    #[test]
    fn converts_text_response() {
        use crate::wire::{
            Candidate, GenerateContentResponse, ResponseContent, ResponsePart, UsageMetadata,
        };
        let resp = GenerateContentResponse {
            response_id: Some("resp_123".to_string()),
            model_version: Some("gemini-2.5-pro".to_string()),
            prompt_feedback: None,
            candidates: vec![Candidate {
                content: Some(ResponseContent {
                    parts: vec![ResponsePart {
                        text: Some("Hello!".to_string()),
                        function_call: None,
                        inline_data: None,
                        file_data: None,
                        thought: None,
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
                total_token_count: Some(15),
                cached_content_token_count: None,
                thoughts_token_count: None,
            }),
        };

        let result = from_api_response(resp).unwrap();
        assert_eq!(result.text(), Some("Hello!".into()));
        assert_eq!(result.finish_reason, Some(FinishReason::Stop));
        assert_eq!(result.id.as_deref(), Some("resp_123"));
        assert_eq!(result.model.as_deref(), Some("gemini-2.5-pro"));
        let usage = result.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(5));
    }

    #[test]
    fn converts_function_call_response() {
        use crate::wire::{Candidate, GenerateContentResponse, ResponseContent, ResponsePart};
        let resp = GenerateContentResponse {
            response_id: Some("resp_123".to_string()),
            model_version: None,
            prompt_feedback: None,
            candidates: vec![Candidate {
                content: Some(ResponseContent {
                    parts: vec![ResponsePart {
                        text: None,
                        function_call: Some(FunctionCallPayload {
                            name: "read_file".to_string(),
                            args: json!({"path": "/tmp/test.txt"}),
                        }),
                        inline_data: None,
                        file_data: None,
                        thought: None,
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: None,
        };

        let result = from_api_response(resp).unwrap();
        // Gemini returns "STOP" but normalization should produce ToolCalls.
        assert_eq!(result.finish_reason, Some(FinishReason::ToolCalls));
        assert!(result.has_tool_calls());
        let calls: Vec<_> = result.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "gemini:resp_123:call:0");
        assert_eq!(calls[0].name, "read_file");
        let args: serde_json::Value = serde_json::from_str(calls[0].arguments).unwrap();
        assert_eq!(args["path"], "/tmp/test.txt");
    }

    #[test]
    fn converts_thought_part_to_reasoning_block() {
        use crate::wire::{Candidate, GenerateContentResponse, ResponseContent, ResponsePart};
        let resp = GenerateContentResponse {
            response_id: None,
            model_version: None,
            prompt_feedback: None,
            candidates: vec![Candidate {
                content: Some(ResponseContent {
                    parts: vec![
                        ResponsePart {
                            text: Some("Let me think...".to_string()),
                            function_call: None,
                            inline_data: None,
                            file_data: None,
                            thought: Some(true),
                        },
                        ResponsePart {
                            text: Some("The answer is 42.".to_string()),
                            function_call: None,
                            inline_data: None,
                            file_data: None,
                            thought: None,
                        },
                    ],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: None,
        };

        let result = from_api_response(resp).unwrap();
        assert_eq!(result.text(), Some("The answer is 42.".into()));
        let reasoning = result.reasoning_text();
        assert_eq!(reasoning, Some("Let me think...".into()));
    }

    #[test]
    fn assigns_deterministic_call_ids_to_parallel_tool_calls() {
        use crate::wire::{Candidate, GenerateContentResponse, ResponseContent, ResponsePart};
        let resp = GenerateContentResponse {
            response_id: Some("resp_123".to_string()),
            model_version: None,
            prompt_feedback: None,
            candidates: vec![Candidate {
                content: Some(ResponseContent {
                    parts: vec![
                        ResponsePart {
                            text: None,
                            function_call: Some(FunctionCallPayload {
                                name: "search".to_string(),
                                args: json!({"query": "rust"}),
                            }),
                            inline_data: None,
                            file_data: None,
                            thought: None,
                        },
                        ResponsePart {
                            text: None,
                            function_call: Some(FunctionCallPayload {
                                name: "search".to_string(),
                                args: json!({"query": "python"}),
                            }),
                            inline_data: None,
                            file_data: None,
                            thought: None,
                        },
                    ],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: None,
        };

        let result = from_api_response(resp).unwrap();
        let calls: Vec<_> = result.tool_calls().collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "gemini:resp_123:call:0");
        assert_eq!(calls[1].id, "gemini:resp_123:call:1");
    }

    #[test]
    fn converts_inline_image_response() {
        use crate::wire::{Candidate, GenerateContentResponse, ResponseContent, ResponsePart};
        let resp = GenerateContentResponse {
            response_id: Some("resp_123".to_string()),
            model_version: None,
            prompt_feedback: None,
            candidates: vec![Candidate {
                content: Some(ResponseContent {
                    parts: vec![ResponsePart {
                        text: None,
                        function_call: None,
                        inline_data: Some(InlineData {
                            mime_type: "image/png".to_string(),
                            data: "iVBORw0KGgoAAAANSUhEUg==".to_string(),
                        }),
                        file_data: None,
                        thought: None,
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: None,
        };

        let result = from_api_response(resp).unwrap();
        assert!(matches!(
            &result.content[..],
            [ContentBlock::Image {
                source: ImageSource::Base64 { media_type, data }
            }] if media_type == "image/png" && data == "iVBORw0KGgoAAAANSUhEUg=="
        ));
    }

    #[test]
    fn converts_assistant_image_to_request_part() {
        let req = ChatRequest::new("gemini-2.0-flash").message(Message::Assistant {
            content: vec![ContentBlock::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png?cache=1".to_string(),
                },
            }],
            name: None,
            extensions: None,
        });

        let api_req = to_api_request(&req).unwrap();
        assert_eq!(api_req.contents[0].role, "model");
        match &api_req.contents[0].parts[0] {
            RequestPart::FileData { file_data } => {
                assert_eq!(file_data.mime_type, "image/png");
                assert_eq!(file_data.file_uri, "https://example.com/cat.png?cache=1");
            }
            other => panic!("expected FileData, got {other:?}"),
        }
    }

    #[test]
    fn rejects_image_url_without_known_extension() {
        let req = ChatRequest::new("gemini-2.0-flash").message(Message::user_multimodal(vec![
            ContentPart::Image {
                source: ImageSource::Url {
                    url: "https://example.com/download?id=1".to_string(),
                },
                detail: None,
            },
        ]));

        let err = to_api_request(&req).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Unsupported(message) if message.contains("cannot infer a supported MIME type"))
        );
    }

    #[test]
    fn empty_candidates_yield_empty_response() {
        use crate::wire::GenerateContentResponse;
        let resp = GenerateContentResponse {
            response_id: Some("resp-empty".into()),
            model_version: Some("gemini-2.5-pro".into()),
            prompt_feedback: None,
            candidates: vec![],
            usage_metadata: Some(UsageMetadata {
                prompt_token_count: Some(3),
                candidates_token_count: Some(0),
                total_token_count: Some(3),
                cached_content_token_count: None,
                thoughts_token_count: None,
            }),
        };

        let response = from_api_response(resp).unwrap();
        assert!(response.content.is_empty());
        assert_eq!(response.finish_reason, None);
        assert_eq!(response.id.as_deref(), Some("resp-empty"));
        assert_eq!(response.model.as_deref(), Some("gemini-2.5-pro"));
        let usage = response.usage.expect("expected usage metadata");
        assert_eq!(usage.input_tokens, Some(3));
        assert_eq!(usage.output_tokens, Some(0));
        assert_eq!(usage.total_tokens, Some(3));
    }

    #[test]
    fn blocked_prompt_maps_to_content_filtered() {
        let resp = GenerateContentResponse {
            response_id: Some("resp-blocked".into()),
            model_version: Some("gemini-2.5-pro".into()),
            prompt_feedback: Some(PromptFeedback {
                block_reason: Some("SAFETY".into()),
                block_reason_message: Some("Prompt blocked by safety filters".into()),
            }),
            candidates: vec![],
            usage_metadata: None,
        };

        let err = from_api_response(resp).unwrap_err();
        assert!(matches!(
            err,
            anyllm::Error::ContentFiltered(message) if message == "Prompt blocked by safety filters"
        ));
    }

    #[test]
    fn parses_gemini_finish_reasons() {
        assert_eq!(parse_finish_reason("STOP"), FinishReason::Stop);
        assert_eq!(parse_finish_reason("MAX_TOKENS"), FinishReason::Length);
        assert_eq!(parse_finish_reason("SAFETY"), FinishReason::ContentFilter);
        assert_eq!(
            parse_finish_reason("MALFORMED_FUNCTION_CALL"),
            FinishReason::ToolCalls
        );
        assert_eq!(
            parse_finish_reason("FINISH_REASON_UNSPECIFIED"),
            FinishReason::Other("FINISH_REASON_UNSPECIFIED".into())
        );
    }
}
