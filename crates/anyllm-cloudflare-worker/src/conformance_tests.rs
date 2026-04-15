#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyllm::{ChatRequest, Message, ResponseFormat, Tool};
    use anyllm_conformance::{
        FixtureDir, assert_error_fixture_eq, assert_json_fixture_eq,
        assert_partial_stream_fixture_eq, assert_response_fixture_eq,
        assert_stream_finish_error_fixture_eq, assert_stream_fixture_eq, load_json_fixture,
        load_text_fixture,
    };
    use serde_json::json;

    fn fixtures() -> FixtureDir {
        FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
    }

    #[test]
    fn request_fixture_matches() {
        let fixtures = fixtures();
        let request = ChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
            .system("You are concise.")
            .message(Message::user("Find rust docs"))
            .temperature(0.2)
            .top_p(0.9)
            .max_tokens(64)
            .seed(42)
            .frequency_penalty(0.1)
            .presence_penalty(0.2)
            .tools(vec![Tool::new(
                "search",
                json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
            )
            .description("Search docs")])
            .response_format(ResponseFormat::JsonSchema {
                name: None,
                schema: json!({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}),
                strict: None,
            });

        let actual = crate::wire::ChatRequest::try_from(&request).unwrap();
        assert_json_fixture_eq(&actual, &fixtures, "request.json");
    }

    #[test]
    fn response_fixture_matches() {
        let fixtures = fixtures();
        let raw = load_json_fixture(&fixtures, "response_raw.json");
        let response: crate::wire::ChatResponse = serde_json::from_value(raw).unwrap();
        let response = anyllm::ChatResponse::try_from(response).unwrap();
        assert_response_fixture_eq(&response, &fixtures, "response_expected.json");
    }

    #[test]
    fn tool_response_fixture_matches() {
        let fixtures = fixtures();
        crate::wire::reset_synthetic_response_ids_for_tests();

        let raw = load_json_fixture(&fixtures, "response_tools_raw.json");
        let response: crate::wire::ChatResponse = serde_json::from_value(raw).unwrap();
        let response = anyllm::ChatResponse::try_from(response).unwrap();
        assert_response_fixture_eq(&response, &fixtures, "response_tools_expected.json");
    }

    #[test]
    fn structured_response_fixture_matches() {
        let fixtures = fixtures();
        let raw = load_json_fixture(&fixtures, "response_json_raw.json");
        let response: crate::wire::ChatResponse = serde_json::from_value(raw).unwrap();
        let response = anyllm::ChatResponse::try_from(response).unwrap();
        assert_response_fixture_eq(&response, &fixtures, "response_json_expected.json");
    }

    #[tokio::test]
    async fn stream_fixture_matches() {
        let fixtures = fixtures();
        let sse = load_text_fixture(&fixtures, "stream.sse");
        assert_stream_fixture_eq(
            crate::streaming::conformance_stream_from_sse_text(&sse),
            &fixtures,
            "stream_events.json",
            "stream_response_expected.json",
        )
        .await;
    }

    #[tokio::test]
    async fn truncated_stream_reports_incomplete_response() {
        let fixtures = fixtures();
        let sse = load_text_fixture(&fixtures, "stream_truncated.sse");

        assert_stream_finish_error_fixture_eq(
            crate::streaming::conformance_stream_from_sse_text(&sse),
            &fixtures,
            "stream_truncated_events.json",
            "stream_truncated_error.json",
        )
        .await;

        assert_partial_stream_fixture_eq(
            crate::streaming::conformance_stream_from_sse_text(&sse),
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
        let auth = crate::error::map_worker_error(worker::Error::RustError(
            "Unauthorized request".to_string(),
        ));
        assert_error_fixture_eq(&auth, &fixtures, "error_auth.json");

        let rate_limited = crate::error::map_worker_error(worker::Error::RustError(
            "Rate limit exceeded".to_string(),
        ));
        assert_error_fixture_eq(&rate_limited, &fixtures, "error_rate_limited.json");

        let timeout = crate::error::map_worker_error(worker::Error::RustError(
            "Worker request timed out".to_string(),
        ));
        assert_error_fixture_eq(&timeout, &fixtures, "error_timeout.json");

        let context = crate::error::map_worker_error(worker::Error::RustError(
            "This model's maximum context length is 32768 tokens".to_string(),
        ));
        assert_error_fixture_eq(&context, &fixtures, "error_context_length.json");

        let model_not_found = crate::error::map_worker_error(worker::Error::RustError(
            "Model not found: @cf/meta/missing".to_string(),
        ));
        assert_error_fixture_eq(&model_not_found, &fixtures, "error_model_not_found.json");

        let content_filtered = crate::error::map_worker_error(worker::Error::RustError(
            "Safety system blocked this response".to_string(),
        ));
        assert_error_fixture_eq(&content_filtered, &fixtures, "error_content_filtered.json");
    }
}
