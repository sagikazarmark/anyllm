#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyllm::{
        ChatRequest, Message, ReasoningConfig, ReasoningEffort, ResponseFormat, Tool, ToolChoice,
    };
    use anyllm_conformance::{
        FixtureDir, assert_error_fixture_eq, assert_json_fixture_eq,
        assert_partial_stream_fixture_eq, assert_response_fixture_eq,
        assert_stream_finish_error_fixture_eq, assert_stream_fixture_eq, load_json_fixture,
        load_text_fixture,
    };
    use serde_json::json;

    use crate::ChatRequestOptions;
    use crate::ChatResponseMetadata;

    fn fixtures() -> FixtureDir {
        FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
    }

    #[test]
    fn request_fixture_matches() {
        let fixtures = fixtures();
        let request = ChatRequest::new("gpt-4o")
            .system("You are a precise assistant.")
            .message(Message::user("Find rust docs"))
            .temperature(0.7)
            .top_p(0.9)
            .max_tokens(256)
            .stop(["END"])
            .frequency_penalty(0.1)
            .presence_penalty(0.2)
            .seed(99)
            .parallel_tool_calls(true)
            .tools(vec![Tool::new(
                "search",
                json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
            )
            .description("Search docs")
            .with_extension("strict", json!(true))])
            .tool_choice(ToolChoice::Specific {
                name: "search".into(),
            })
            .response_format(ResponseFormat::JsonSchema {
                name: Some("answer".into()),
                schema: json!({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}),
                strict: Some(true),
            })
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: Some(4096),
                effort: Some(ReasoningEffort::High),
            })
            .with_option(ChatRequestOptions {
                user: Some("user_123".into()),
                service_tier: Some("flex".into()),
                store: Some(true),
                metadata: Some(serde_json::Map::from_iter([(
                    "trace_id".into(),
                    json!("trace-123"),
                )])),
                reasoning_effort: None,
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
        assert_eq!(
            response.metadata.get::<ChatResponseMetadata>(),
            Some(&ChatResponseMetadata {
                system_fingerprint: Some("fp_123".into()),
            })
        );
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
                "error": {
                    "type": "invalid_request_error",
                    "message": "Incorrect API key provided",
                    "code": "invalid_api_key"
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
                "error": {
                    "type": "rate_limit_error",
                    "message": "Rate limit reached"
                }
            })
            .to_string(),
            Some("req-openai-429".into()),
            Some(std::time::Duration::from_secs(30)),
        );
        assert_error_fixture_eq(&rate_limited, &fixtures, "error_rate_limited.json");

        let context = crate::error::conformance_map_http_error(
            400,
            &serde_json::json!({
                "error": {
                    "type": "invalid_request_error",
                    "message": "This model's maximum context length is 128000 tokens"
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
                "error": {
                    "type": "invalid_request_error",
                    "message": "The model `gpt-unknown` does not exist",
                    "code": "model_not_found"
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
}
