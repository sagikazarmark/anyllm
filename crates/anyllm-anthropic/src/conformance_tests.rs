#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyllm::{ChatRequest, Message, ReasoningConfig, Tool, ToolChoice};
    use anyllm_conformance::{
        FixtureDir, assert_error_fixture_eq, assert_json_fixture_eq,
        assert_partial_stream_fixture_eq, assert_response_fixture_eq,
        assert_stream_finish_error_fixture_eq, assert_stream_fixture_eq, load_json_fixture,
        load_text_fixture,
    };
    use serde_json::json;

    use crate::ChatRequestOptions;

    fn fixtures() -> FixtureDir {
        FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
    }

    #[test]
    fn request_fixture_matches() {
        let fixtures = fixtures();
        let request = ChatRequest::new("claude-sonnet-4-20250514")
            .system(
                anyllm::SystemPrompt::new("You are a careful assistant.")
                    .with_option(crate::CacheControl::ephemeral()),
            )
            .message(Message::user("Find the answer"))
            .tools(vec![
                Tool::new(
                    "search",
                    json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
                )
                .description("Search docs"),
            ])
            .tool_choice(ToolChoice::Specific {
                name: "search".into(),
            })
            .temperature(0.3)
            .top_p(0.8)
            .stop(["END"])
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: Some(2048),
                effort: None,
            })
            .with_option(ChatRequestOptions {
                top_k: Some(12),
                metadata: Some(serde_json::Map::from_iter([(
                    "trace_id".into(),
                    json!("trace-anthropic-123"),
                )])),
                anthropic_beta: Vec::new(),
                budget_tokens: None,
            });

        let actual = crate::wire::conformance_to_api_request(&request, false).unwrap();
        assert_json_fixture_eq(&actual, &fixtures, "request.json");
    }

    #[test]
    fn response_fixture_matches() {
        let fixtures = fixtures();
        let raw = load_json_fixture(&fixtures, "response_raw.json");
        let response = crate::wire::conformance_from_api_response_value(raw).unwrap();
        assert_response_fixture_eq(&response, &fixtures, "response_expected.json");
    }

    #[tokio::test]
    async fn stream_fixture_matches() {
        let fixtures = fixtures();
        let sse = load_text_fixture(&fixtures, "stream.sse");
        let stream = crate::conformance_stream_from_sse_text(&sse);
        assert_stream_fixture_eq(
            stream,
            &fixtures,
            "stream_events.json",
            "stream_response_expected.json",
        )
        .await;
    }

    #[tokio::test]
    async fn truncated_stream_fixture_reports_strict_error_and_partial_recovery() {
        let fixtures = fixtures();
        let sse = load_text_fixture(&fixtures, "stream_truncated.sse");

        assert_stream_finish_error_fixture_eq(
            crate::conformance_stream_from_sse_text(&sse),
            &fixtures,
            "stream_truncated_events.json",
            "stream_truncated_error.json",
        )
        .await;

        assert_partial_stream_fixture_eq(
            crate::conformance_stream_from_sse_text(&sse),
            &fixtures,
            "stream_truncated_events.json",
            "stream_truncated_response_expected.json",
            "stream_truncated_completeness.json",
        )
        .await;
    }

    #[test]
    fn error_fixtures_match() {
        let fixtures = fixtures();
        let auth = crate::error::conformance_map_http_error(
            401,
            &serde_json::json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "invalid x-api-key"
                }
            })
            .to_string(),
            None,
            None,
        );
        assert_error_fixture_eq(&auth, &fixtures, "error_auth.json");

        let rate_limited = crate::error::conformance_map_http_error(
            429,
            &serde_json::json!({
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "too many requests"
                }
            })
            .to_string(),
            Some("req-anthropic-429".into()),
            None,
        );
        assert_error_fixture_eq(&rate_limited, &fixtures, "error_rate_limited.json");

        let context = crate::error::conformance_map_http_error(
            400,
            &serde_json::json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: 200000 tokens"
                }
            })
            .to_string(),
            None,
            None,
        );
        assert_error_fixture_eq(&context, &fixtures, "error_context_length.json");

        let model_not_found = crate::error::conformance_map_http_error(
            404,
            &serde_json::json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "model not found"
                }
            })
            .to_string(),
            None,
            None,
        );
        assert_error_fixture_eq(&model_not_found, &fixtures, "error_model_not_found.json");

        let timeout = crate::error::conformance_map_timeout_transport_error();
        assert_error_fixture_eq(&timeout, &fixtures, "error_timeout.json");
    }

    #[test]
    fn structured_response_format_is_rejected() {
        let request = ChatRequest::new("claude-sonnet-4-20250514")
            .message(Message::user("Return JSON"))
            .response_format(anyllm::ResponseFormat::Json);

        let err = crate::wire::CreateMessageRequest::try_from(&request).unwrap_err();
        assert!(
            matches!(err, anyllm::Error::Unsupported(message) if message.contains("response_format"))
        );
    }
}
