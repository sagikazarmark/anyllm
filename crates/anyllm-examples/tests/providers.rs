use anyllm::prelude::*;
use anyllm::{Error, Result, StreamCollector, Tool, ToolChoice};
use anyllm_examples::{
    ALL_PROVIDER_KINDS, EmbeddingProviderKind, LoadedEmbeddingProvider, ProviderKind,
    load_embedding_provider, load_provider,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

mod support;

#[derive(Debug, Deserialize, JsonSchema)]
struct LiveExtract {
    token: String,
    summary: String,
}

#[derive(Debug, Deserialize)]
struct LookupFactArgs {
    topic: String,
}

fn provider_is_selected(selection: &str, kind: ProviderKind) -> bool {
    support::selection_includes(selection, kind.as_str())
}

fn primary_credential_is_set(kind: ProviderKind) -> bool {
    std::env::var(kind.credential_env())
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

fn load_live_provider(
    selection: &str,
    kind: ProviderKind,
) -> Result<Option<anyllm_examples::LoadedProvider>> {
    if !provider_is_selected(selection, kind) {
        return Ok(None);
    }

    if !primary_credential_is_set(kind) {
        if support::selection_uses_configured_alias(selection) {
            eprintln!(
                "skipping {} live checks; {} is not set",
                kind.as_str(),
                kind.credential_env()
            );
            return Ok(None);
        }

        return Err(Error::InvalidRequest(format!(
            "{} selected by ANYLLM_LIVE_PROVIDER but {} is not set",
            kind.as_str(),
            kind.credential_env()
        )));
    }

    load_provider(kind).map(Some)
}

fn lookup_fact_tool() -> Tool {
    Tool::new(
        "lookup_fact",
        json!({
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to look up"
                }
            },
            "required": ["topic"],
            "additionalProperties": false
        }),
    )
    .description("Look up a short factual answer for the requested topic.")
}

async fn run_live_chat(target: &anyllm_examples::LoadedProvider) -> Result<()> {
    let response = target
        .provider
        .chat(
            &ChatRequest::new(&target.model)
                .system("Reply in one short sentence and include the exact token live-chat-ok.")
                .user("Confirm that this provider smoke test is working."),
        )
        .await?;

    support::assert_contains_token(&response.text_or_empty(), "live-chat-ok");
    Ok(())
}

async fn run_live_stream(target: &anyllm_examples::LoadedProvider) -> Result<()> {
    let request = ChatRequest::new(&target.model)
        .system("Reply in one short sentence and include the exact token live-stream-ok.")
        .user("Confirm that this streaming smoke test is working.");

    let mut stream = target.provider.chat_stream(&request).await?;
    let mut collector = StreamCollector::new();
    let mut saw_text_delta = false;

    while let Some(item) = stream.next().await {
        let event = item?;
        saw_text_delta |= matches!(event, StreamEvent::TextDelta { .. });
        collector.push(event)?;
    }

    let response = collector.finish()?;
    assert!(saw_text_delta, "expected at least one text delta");
    support::assert_contains_token(&response.text_or_empty(), "live-stream-ok");
    Ok(())
}

async fn run_live_tool_roundtrip(target: &anyllm_examples::LoadedProvider) -> Result<()> {
    let mut request = ChatRequest::new(&target.model)
        .system(
            "When tools are available and required, call the lookup_fact tool first. After you receive the tool result, answer with that result in one short sentence.",
        )
        .user("What color is the sky?")
        .tools(vec![lookup_fact_tool()])
        .tool_choice(ToolChoice::Required);

    let first = target.provider.chat(&request).await?;
    assert!(first.has_tool_calls(), "expected at least one tool call");

    request.push_message(first.to_assistant_message());
    for call in first.tool_calls() {
        assert_eq!(call.name, "lookup_fact", "unexpected live tool name");

        let args: LookupFactArgs = call.parse_arguments()?;
        let normalized_topic = args.topic.trim().to_ascii_lowercase();
        assert!(
            normalized_topic == "sky" || normalized_topic == "the sky",
            "expected lookup_fact topic to target the sky, got: {}",
            args.topic
        );

        request.push_tool_result(call, "The sky is blue. live-tool-ok.");
    }

    let second = target.provider.chat(&request).await?;
    support::assert_contains_token(&second.text_or_empty(), "live-tool-ok");
    Ok(())
}

fn embedding_kind_for(kind: ProviderKind) -> Option<EmbeddingProviderKind> {
    match kind {
        ProviderKind::OpenAI => Some(EmbeddingProviderKind::OpenAI),
        ProviderKind::Gemini => Some(EmbeddingProviderKind::Gemini),
        ProviderKind::Anthropic => None,
    }
}

async fn run_live_embed(target: &LoadedEmbeddingProvider) -> Result<()> {
    let inputs = [
        "anyllm live embedding smoke input A".to_string(),
        "anyllm live embedding smoke input B".to_string(),
    ];
    let request = EmbeddingRequest::new(&target.model).inputs(inputs.iter().cloned());
    let response = target.provider.embed(&request).await?;

    assert_eq!(
        response.embeddings.len(),
        inputs.len(),
        "expected embedding count to match input count"
    );
    let dim = response.embeddings[0].len();
    assert!(dim > 0, "expected non-empty embedding vector");
    for vector in &response.embeddings {
        assert_eq!(
            vector.len(),
            dim,
            "all embedding vectors should share the same length"
        );
    }
    Ok(())
}

async fn run_live_extract(target: &anyllm_examples::LoadedProvider) -> Result<()> {
    let request = ChatRequest::new(&target.model)
        .system(
            "Return structured data with token set to the exact string live-extract-ok and summary set to one short sentence.",
        )
        .user("Confirm that this structured extraction smoke test is working.");

    let extracted: Extracted<LiveExtract> = target.provider.extract(&request).await?;
    assert_eq!(extracted.value.token, "live-extract-ok");
    assert!(
        !extracted.value.summary.trim().is_empty(),
        "expected structured summary to be populated"
    );
    Ok(())
}

#[tokio::test]
async fn live_http_providers_smoke() -> Result<()> {
    let Some(selection) = support::live_selection() else {
        eprintln!(
            "skipping HTTP provider live checks; set ANYLLM_LIVE_PROVIDER=openai|anthropic|gemini|all"
        );
        return Ok(());
    };

    support::validate_live_selection(&selection)?;
    support::ensure_selection_has_runnable_target(&selection)?;

    let mut executed = 0usize;
    for kind in ALL_PROVIDER_KINDS {
        let Some(target) = load_live_provider(&selection, kind)? else {
            continue;
        };

        eprintln!(
            "running live HTTP smoke checks for provider={} model={}",
            target.kind.as_str(),
            target.model
        );
        run_live_chat(&target).await?;
        run_live_stream(&target).await?;
        run_live_tool_roundtrip(&target).await?;
        run_live_extract(&target).await?;

        if let Some(embedding_kind) = embedding_kind_for(target.kind) {
            let embedding_target = load_embedding_provider(embedding_kind)?;
            eprintln!(
                "running live embedding smoke check for provider={} model={}",
                embedding_target.kind.as_str(),
                embedding_target.model
            );
            run_live_embed(&embedding_target).await?;
        }

        executed += 1;
    }

    if executed == 0 {
        eprintln!(
            "no configured HTTP providers matched ANYLLM_LIVE_PROVIDER={selection}; another selected target must satisfy the live gate"
        );
    }

    Ok(())
}
