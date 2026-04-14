#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyllm::{ChatRequest, Message, ReasoningConfig, ResponseFormat, Tool, ToolChoice};
    use anyllm_conformance::{
        FixtureDir, assert_error_fixture_eq, assert_json_fixture_eq,
        assert_partial_stream_fixture_eq, assert_response_fixture_eq, assert_stream_fixture_eq,
        load_json_fixture, load_text_fixture,
    };
    use serde_json::json;

    use crate::ChatRequestOptions;

    fn fixtures() -> FixtureDir {
        FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
    }

    #[test]
    fn request_fixture_matches() {
        let fixtures = fixtures();
        let request = ChatRequest::new("gemini-2.5-pro")
            .system("You are a precise assistant.")
            .message(Message::user("Find rust docs"))
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
            .temperature(0.4)
            .top_p(0.8)
            .max_tokens(256)
            .stop(["END"])
            .response_format(ResponseFormat::JsonSchema {
                name: None,
                schema: json!({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}),
                strict: None,
            })
            .reasoning(ReasoningConfig {
                enabled: true,
                budget_tokens: Some(1024),
                effort: None,
            })
            .with_option(ChatRequestOptions {
                top_k: Some(7),
                candidate_count: Some(1),
                response_mime_type: Some("application/json".into()),
                cached_content: Some("cached/abc".into()),
                thinking_budget: None,
            });

        let actual = crate::wire::conformance_to_api_request(&request).unwrap();
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
    async fn truncated_stream_fixture_collects_after_eof_finalization() {
        let fixtures = fixtures();
        let sse = load_text_fixture(&fixtures, "stream_truncated.sse");

        assert_stream_fixture_eq(
            crate::conformance_stream_from_sse_text(&sse),
            &fixtures,
            "stream_truncated_events.json",
            "stream_truncated_response_expected.json",
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
                    "code": 401,
                    "message": "API key not valid",
                    "status": "UNAUTHENTICATED"
                }
            })
            .to_string(),
            None,
        );
        assert_error_fixture_eq(&auth, &fixtures, "error_auth.json");

        let rate_limited = crate::error::conformance_map_http_error(
            429,
            &serde_json::json!({
                "error": {
                    "code": 429,
                    "message": "Quota exceeded",
                    "status": "RESOURCE_EXHAUSTED"
                }
            })
            .to_string(),
            Some(std::time::Duration::from_secs(30)),
        );
        assert_error_fixture_eq(&rate_limited, &fixtures, "error_rate_limited.json");

        let context = crate::error::conformance_map_http_error(
            400,
            &serde_json::json!({
                "error": {
                    "code": 400,
                    "message": "This model's maximum context length is exceeded",
                    "status": "INVALID_ARGUMENT"
                }
            })
            .to_string(),
            None,
        );
        assert_error_fixture_eq(&context, &fixtures, "error_context_length.json");

        let model_not_found = crate::error::conformance_map_http_error(
            404,
            &serde_json::json!({
                "error": {
                    "code": 404,
                    "message": "Model not found",
                    "status": "NOT_FOUND"
                }
            })
            .to_string(),
            None,
        );
        assert_error_fixture_eq(&model_not_found, &fixtures, "error_model_not_found.json");

        let timeout = crate::error::conformance_map_timeout_transport_error();
        assert_error_fixture_eq(&timeout, &fixtures, "error_timeout.json");
    }
}
