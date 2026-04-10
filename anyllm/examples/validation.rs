use anyllm::prelude::*;
use serde_json::json;

fn weather_tool() -> Tool {
    Tool::new(
        "lookup_weather",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"],
            "additionalProperties": false
        }),
    )
    .description("Look up a fake weather report.")
}

fn build_request() -> ChatRequest {
    ChatRequest::new("demo-model")
        .user("What is the weather in San Francisco?")
        .tools([weather_tool()])
        .tool_choice(ToolChoice::Required)
}

fn build_provider() -> MockProvider {
    MockProvider::with_text("Sunny enough.")
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported)
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let request = build_request();

    let strict = ValidatingChatProvider::new(build_provider()).with_mode(ValidationMode::Strict);
    let err = strict.chat(&request).await.unwrap_err();
    println!("strict validation rejected request: {err}");
    println!(
        "strict provider call count: {}",
        strict.into_inner().call_count()
    );

    let permissive =
        ValidatingChatProvider::new(build_provider()).with_mode(ValidationMode::Permissive);
    let response = permissive.chat(&request).await?;
    println!(
        "permissive validation allowed dispatch: {}",
        response.text_or_empty()
    );
    println!(
        "permissive provider call count: {}",
        permissive.into_inner().call_count()
    );

    Ok(())
}
