use anyllm::{ChatRequest, ChatResponse, Result};

use crate::{ChatRequestOptions, ChatResponseMetadata};
pub(crate) use anyllm_openai_compat::{ChatCompletionRequest, ChatCompletionResponse};
use anyllm_openai_compat::{
    ApiMessage, RequestOptions as CompatRequestOptions,
    from_api_response as compat_from_api_response, to_chat_completion_request,
};

/// Convert an anyllm `ChatRequest` to an OpenAI `ChatCompletionRequest`.
pub(crate) fn to_api_request(request: &ChatRequest, stream: bool) -> Result<ChatCompletionRequest> {
    let provider_options = request.option::<ChatRequestOptions>();
    let mut compat_options = CompatRequestOptions::default();
    if let Some(options) = provider_options {
        compat_options.user = options.user.clone();
        compat_options.service_tier = options.service_tier.clone();
        compat_options.store = options.store;
        compat_options.metadata = options.metadata.clone();
        compat_options.reasoning_effort = options.reasoning_effort;
    }

    let mut api_request = to_chat_completion_request(request, stream, &compat_options)?;

    // Emit req.system as one role: "system" wire message per prompt, in order,
    // before any req.messages entries. Empty-content prompts are skipped so
    // we never put empty system messages on the wire. Typed SystemOptions
    // entries are silently ignored — no OpenAI-specific options are defined
    // in V1.
    let mut system_messages: Vec<ApiMessage> = Vec::with_capacity(request.system.len());
    for prompt in &request.system {
        if prompt.content.is_empty() {
            continue;
        }
        system_messages.push(ApiMessage {
            role: "system".to_string(),
            content: Some(serde_json::json!(prompt.content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }
    if !system_messages.is_empty() {
        system_messages.append(&mut api_request.messages);
        api_request.messages = system_messages;
    }

    Ok(api_request)
}

/// Convert an OpenAI `ChatCompletionResponse` to an anyllm `ChatResponse`.
pub(crate) fn from_api_response(response: ChatCompletionResponse) -> Result<ChatResponse> {
    compat_from_api_response(response, |response, metadata| {
        if response.system_fingerprint.is_some() {
            metadata.insert(ChatResponseMetadata {
                system_fingerprint: response.system_fingerprint.clone(),
            });
        }
    })
}

#[cfg(test)]
pub(crate) fn conformance_to_api_request(
    request: &ChatRequest,
    stream: bool,
) -> Result<serde_json::Value> {
    serde_json::to_value(to_api_request(request, stream)?).map_err(anyllm::Error::from)
}

#[cfg(test)]
pub(crate) fn conformance_from_api_response_value(
    value: serde_json::Value,
) -> Result<ChatResponse> {
    let response: ChatCompletionResponse =
        serde_json::from_value(value).map_err(anyllm::Error::from)?;
    from_api_response(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyllm::{
        ContentBlock, ContentPart, FinishReason, ImageSource, Message, ReasoningConfig,
        ReasoningEffort, ResponseFormat, Tool, ToolChoice,
    };
    use anyllm_openai_compat::{ChatCompletionResponse, parse_finish_reason};
    use serde_json::json;

    fn response(value: serde_json::Value) -> ChatCompletionResponse {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn converts_simple_user_message() {
        let req = ChatRequest::new("gpt-4o").message(Message::user("Hello!"));
        let api_req = to_api_request(&req, false).unwrap();

        assert_eq!(api_req.model, "gpt-4o");
        assert!(!api_req.stream);
        assert_eq!(api_req.messages.len(), 1);
        assert_eq!(api_req.messages[0].role, "user");
        assert_eq!(api_req.messages[0].content, Some(json!("Hello!")));
    }

    #[test]
    fn system_messages_stay_in_array() {
        let req = ChatRequest::new("gpt-4o")
            .system("You are helpful")
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();

        assert_eq!(api_req.messages.len(), 2);
        assert_eq!(api_req.messages[0].role, "system");
        assert_eq!(api_req.messages[0].content, Some(json!("You are helpful")));
        assert_eq!(api_req.messages[1].role, "user");
    }

    #[test]
    fn splits_assistant_content_blocks_into_text_and_tool_calls() {
        let req = ChatRequest::new("gpt-4o").message(Message::Assistant {
            content: vec![
                ContentBlock::Text {
                    text: "Let me check.".into(),
                },
                ContentBlock::ToolCall {
                    id: "call_abc".into(),
                    name: "read_file".into(),
                    arguments: r#"{"path":"/tmp/test.txt"}"#.into(),
                },
            ],
            name: None,
            extensions: None,
        });

        let api_req = to_api_request(&req, false).unwrap();
        let msg = &api_req.messages[0];

        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, Some(json!("Let me check.")));
        let tool_calls = msg.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc");
        assert_eq!(tool_calls[0].call_type, "function");
        assert_eq!(tool_calls[0].function.name, "read_file");
        assert_eq!(
            tool_calls[0].function.arguments,
            r#"{"path":"/tmp/test.txt"}"#
        );
    }

    #[test]
    fn assistant_with_only_tool_calls_has_null_content() {
        let req = ChatRequest::new("gpt-4o").message(Message::Assistant {
            content: vec![ContentBlock::ToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: "{}".into(),
            }],
            name: None,
            extensions: None,
        });

        let api_req = to_api_request(&req, false).unwrap();
        let msg = &api_req.messages[0];

        assert!(msg.content.is_none());
        assert_eq!(msg.tool_calls.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn drops_reasoning_blocks_from_assistant_message() {
        let req = ChatRequest::new("gpt-4o").message(Message::Assistant {
            content: vec![
                ContentBlock::Reasoning {
                    text: "thinking...".into(),
                    signature: Some("sig".into()),
                },
                ContentBlock::Text {
                    text: "Answer".into(),
                },
            ],
            name: None,
            extensions: None,
        });

        let api_req = to_api_request(&req, false).unwrap();
        let msg = &api_req.messages[0];

        assert_eq!(msg.content, Some(json!("Answer")));
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn converts_tool_result_message() {
        let req = ChatRequest::new("gpt-4o").message(Message::tool_result(
            "call_abc",
            "read_file",
            "file contents",
        ));

        let api_req = to_api_request(&req, false).unwrap();
        let msg = &api_req.messages[0];

        assert_eq!(msg.role, "tool");
        assert_eq!(msg.content, Some(json!("file contents")));
        assert_eq!(msg.tool_call_id, Some("call_abc".to_string()));
    }

    #[test]
    fn tool_results_are_separate_messages() {
        let req = ChatRequest::new("gpt-4o")
            .message(Message::tool_result("call_1", "tool_a", "result one"))
            .message(Message::tool_result("call_2", "tool_b", "result two"));

        let api_req = to_api_request(&req, false).unwrap();

        // Unlike Anthropic, OpenAI keeps each tool result as its own message
        assert_eq!(api_req.messages.len(), 2);
        assert_eq!(api_req.messages[0].tool_call_id, Some("call_1".into()));
        assert_eq!(api_req.messages[1].tool_call_id, Some("call_2".into()));
    }

    #[test]
    fn maps_tools_with_strict() {
        let req = ChatRequest::new("gpt-4o")
            .tools(vec![
                Tool::new("search", json!({"type": "object"}))
                    .description("Search the web")
                    .with_extension("strict", json!(true)),
            ])
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();
        let tools = api_req.tools.unwrap();

        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "search");
        assert_eq!(tools[0].function.strict, Some(true));
    }

    #[test]
    fn tool_choice_auto() {
        let req = ChatRequest::new("gpt-4o")
            .tools(vec![Tool::new("t", json!({})).description("d")])
            .tool_choice(ToolChoice::Auto)
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();
        assert_eq!(api_req.tool_choice, Some(json!("auto")));
    }

    #[test]
    fn tool_choice_required() {
        let req = ChatRequest::new("gpt-4o")
            .tools(vec![Tool::new("t", json!({})).description("d")])
            .tool_choice(ToolChoice::Required)
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();
        assert_eq!(api_req.tool_choice, Some(json!("required")));
    }

    #[test]
    fn tool_choice_disabled_sends_none_string() {
        let req = ChatRequest::new("gpt-4o")
            .tools(vec![Tool::new("t", json!({})).description("d")])
            .tool_choice(ToolChoice::Disabled)
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();
        // Unlike Anthropic which omits tools, OpenAI sends "none"
        assert_eq!(api_req.tool_choice, Some(json!("none")));
        assert!(api_req.tools.is_some()); // tools are still present
    }

    #[test]
    fn tool_choice_specific() {
        let req = ChatRequest::new("gpt-4o")
            .tools(vec![Tool::new("read_file", json!({})).description("d")])
            .tool_choice(ToolChoice::Specific {
                name: "read_file".into(),
            })
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();
        assert_eq!(
            api_req.tool_choice,
            Some(json!({"type": "function", "function": {"name": "read_file"}}))
        );
    }

    #[test]
    fn empty_tools_vec_maps_to_none() {
        let req = ChatRequest::new("gpt-4o")
            .tools(vec![])
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();
        assert!(api_req.tools.is_none());
    }

    #[test]
    fn response_format_json() {
        let req = ChatRequest::new("gpt-4o")
            .response_format(ResponseFormat::Json)
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();
        assert_eq!(
            api_req.response_format,
            Some(json!({"type": "json_object"}))
        );
    }

    #[test]
    fn response_format_json_schema() {
        let req = ChatRequest::new("gpt-4o")
            .response_format(ResponseFormat::JsonSchema {
                name: Some("person".into()),
                schema: json!({"type": "object", "properties": {"name": {"type": "string"}}}),
                strict: Some(true),
            })
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();
        let rf = api_req.response_format.unwrap();
        assert_eq!(rf["type"], "json_schema");
        assert_eq!(rf["json_schema"]["name"], "person");
        assert_eq!(rf["json_schema"]["strict"], true);
    }

    #[test]
    fn reasoning_effort_from_explicit_effort() {
        let req = ChatRequest::new("o3")
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: None,
                effort: Some(ReasoningEffort::High),
            })
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();
        assert_eq!(api_req.reasoning_effort, Some("high".to_string()));
    }

    #[test]
    fn reasoning_effort_derived_from_budget() {
        // Low: < 1024
        let req = ChatRequest::new("o3")
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: Some(512),
                effort: None,
            })
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req, false).unwrap();
        assert_eq!(api_req.reasoning_effort, Some("low".to_string()));

        // Medium: < 4096
        let req = ChatRequest::new("o3")
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: Some(2048),
                effort: None,
            })
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req, false).unwrap();
        assert_eq!(api_req.reasoning_effort, Some("medium".to_string()));

        // High: >= 4096
        let req = ChatRequest::new("o3")
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: Some(8192),
                effort: None,
            })
            .message(Message::user("Hi"));
        let api_req = to_api_request(&req, false).unwrap();
        assert_eq!(api_req.reasoning_effort, Some("high".to_string()));
    }

    #[test]
    fn reasoning_disabled_maps_to_none() {
        let req = ChatRequest::new("o3")
            .reasoning(ReasoningConfig {
                enabled: false,
                budget_tokens: Some(8192),
                effort: Some(ReasoningEffort::High),
            })
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();
        assert!(api_req.reasoning_effort.is_none());
    }

    #[test]
    fn reasoning_enabled_no_effort_no_budget_omitted() {
        let req = ChatRequest::new("o3")
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: None,
                effort: None,
            })
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();
        assert!(api_req.reasoning_effort.is_none());
    }

    #[test]
    fn converts_multimodal_user_with_image_url() {
        let req = ChatRequest::new("gpt-4o").message(Message::user_multimodal(vec![
            ContentPart::text("What's in this image?"),
            ContentPart::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png".into(),
                },
                detail: Some("high".into()),
            },
        ]));

        let api_req = to_api_request(&req, false).unwrap();
        let content = api_req.messages[0].content.as_ref().unwrap();
        let parts = content.as_array().unwrap();

        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "What's in this image?");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "https://example.com/cat.png");
        assert_eq!(parts[1]["image_url"]["detail"], "high");
    }

    #[test]
    fn converts_base64_image_to_data_uri() {
        let req = ChatRequest::new("gpt-4o").message(Message::user_multimodal(vec![
            ContentPart::Image {
                source: ImageSource::Base64 {
                    media_type: "image/png".into(),
                    data: "iVBORw0KGgo=".into(),
                },
                detail: None,
            },
        ]));

        let api_req = to_api_request(&req, false).unwrap();
        let content = api_req.messages[0].content.as_ref().unwrap();
        let parts = content.as_array().unwrap();

        assert_eq!(parts[0]["type"], "image_url");
        assert_eq!(
            parts[0]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo="
        );
        assert!(parts[0]["image_url"].get("detail").is_none());
    }

    #[test]
    fn streaming_sets_stream_options() {
        let req = ChatRequest::new("gpt-4o").message(Message::user("Hi"));
        let api_req = to_api_request(&req, true).unwrap();

        assert!(api_req.stream);
        let opts = api_req.stream_options.unwrap();
        assert!(opts.include_usage);
    }

    #[test]
    fn non_streaming_omits_stream_options() {
        let req = ChatRequest::new("gpt-4o").message(Message::user("Hi"));
        let api_req = to_api_request(&req, false).unwrap();

        assert!(!api_req.stream);
        assert!(api_req.stream_options.is_none());
    }

    #[test]
    fn maps_all_optional_fields() {
        let req = ChatRequest::new("gpt-4o")
            .temperature(0.7)
            .top_p(0.9)
            .max_tokens(4096)
            .stop(["END"])
            .frequency_penalty(0.5)
            .presence_penalty(0.3)
            .seed(42)
            .parallel_tool_calls(true)
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();

        assert_eq!(api_req.temperature, Some(0.7));
        assert_eq!(api_req.top_p, Some(0.9));
        assert_eq!(api_req.max_tokens, Some(4096));
        assert_eq!(api_req.stop, Some(vec!["END".to_string()]));
        assert_eq!(api_req.frequency_penalty, Some(0.5));
        assert_eq!(api_req.presence_penalty, Some(0.3));
        assert_eq!(api_req.seed, Some(42));
        assert_eq!(api_req.parallel_tool_calls, Some(true));
    }

    #[test]
    fn provider_options_only_set_provider_specific_transport_fields() {
        let req = ChatRequest::new("gpt-4o")
            .seed(42)
            .parallel_tool_calls(true)
            .with_option(ChatRequestOptions {
                user: Some("user-123".into()),
                service_tier: Some("flex".into()),
                store: Some(true),
                metadata: Some(serde_json::Map::from_iter([(
                    "team".to_string(),
                    json!("agents"),
                )])),
                reasoning_effort: None,
            })
            .message(Message::user("Hi"));

        let api_req = to_api_request(&req, false).unwrap();

        assert_eq!(api_req.seed, Some(42));
        assert_eq!(api_req.parallel_tool_calls, Some(true));
        assert_eq!(api_req.user.as_deref(), Some("user-123"));
        assert_eq!(api_req.service_tier.as_deref(), Some("flex"));
        assert_eq!(api_req.store, Some(true));
        assert_eq!(api_req.metadata.as_ref().unwrap()["team"], "agents");
    }

    #[test]
    fn converts_text_response() {
        let resp = response(json!({
            "id": "chatcmpl-abc",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "model": "gpt-4o-2024-05-13",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "system_fingerprint": "fp_abc"
        }));

        let result = from_api_response(resp).unwrap();

        assert_eq!(result.text(), Some("Hello!".into()));
        assert_eq!(result.finish_reason, Some(FinishReason::Stop));
        assert_eq!(result.model, Some("gpt-4o-2024-05-13".into()));
        assert_eq!(result.id, Some("chatcmpl-abc".into()));
        let usage = result.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(15));
        let metadata = result.metadata.get::<ChatResponseMetadata>().unwrap();
        assert_eq!(metadata.system_fingerprint.as_deref(), Some("fp_abc"));
    }

    #[test]
    fn converts_tool_call_response() {
        let resp = response(json!({
            "id": "chatcmpl-xyz",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": r#"{"path":"/etc/hosts"}"#
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "total_tokens": 80
            }
        }));

        let result = from_api_response(resp).unwrap();

        assert_eq!(result.finish_reason, Some(FinishReason::ToolCalls));
        assert!(result.has_tool_calls());
        assert_eq!(result.text(), Some("Let me check.".into()));
        let calls: Vec<_> = result.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].name, "read_file");
        let args: serde_json::Value = serde_json::from_str(calls[0].arguments).unwrap();
        assert_eq!(args["path"], "/etc/hosts");
    }

    #[test]
    fn converts_response_with_usage_details() {
        let resp = response(json!({
            "id": "chatcmpl-detail",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Answer"
                },
                "finish_reason": "stop"
            }],
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {
                    "cached_tokens": 80
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 20
                }
            }
        }));

        let result = from_api_response(resp).unwrap();
        let usage = result.usage.unwrap();
        assert_eq!(usage.cached_input_tokens, Some(80));
        assert_eq!(usage.reasoning_tokens, Some(20));
    }

    #[test]
    fn converts_refusal_as_text() {
        let resp = response(json!({
            "id": "chatcmpl-refuse",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "refusal": "I cannot help with that."
                },
                "finish_reason": "stop"
            }],
            "model": "gpt-4o"
        }));

        let result = from_api_response(resp).unwrap();
        assert_eq!(result.text(), Some("I cannot help with that.".into()));
    }

    #[test]
    fn errors_on_empty_choices() {
        let resp = response(json!({
            "id": "chatcmpl-empty",
            "choices": [],
            "model": "gpt-4o"
        }));

        let err = from_api_response(resp).unwrap_err();
        assert!(matches!(err, anyllm::Error::Provider { .. }));
    }

    #[test]
    fn parses_known_finish_reasons() {
        assert_eq!(parse_finish_reason("stop"), FinishReason::Stop);
        assert_eq!(parse_finish_reason("length"), FinishReason::Length);
        assert_eq!(parse_finish_reason("tool_calls"), FinishReason::ToolCalls);
        assert_eq!(
            parse_finish_reason("content_filter"),
            FinishReason::ContentFilter
        );
    }

    #[test]
    fn parses_unknown_finish_reason_as_other() {
        assert_eq!(
            parse_finish_reason("something_new"),
            FinishReason::Other("something_new".into())
        );
    }

    #[test]
    fn req_system_emits_system_messages_at_front() {
        use anyllm::SystemPrompt;

        let mut req = ChatRequest::new("gpt-4o").user("hi");
        req.system.push(SystemPrompt::new("A"));
        req.system.push(SystemPrompt::new("B"));

        let api_req = to_api_request(&req, false).unwrap();
        assert_eq!(api_req.messages[0].role, "system");
        assert_eq!(api_req.messages[0].content, Some(json!("A")));
        assert_eq!(api_req.messages[1].role, "system");
        assert_eq!(api_req.messages[1].content, Some(json!("B")));
        assert_eq!(api_req.messages[2].role, "user");
    }

    #[test]
    fn req_system_empty_emits_no_system_messages() {
        let req = ChatRequest::new("gpt-4o").user("hi");
        let api_req = to_api_request(&req, false).unwrap();
        assert_eq!(api_req.messages[0].role, "user");
    }
}
