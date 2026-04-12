use anyllm::prelude::*;
use anyllm::{CapabilitySupport, ChatCapability, Error, ExtractError, ExtractingProvider};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize, JsonSchema)]
struct Review {
    title: String,
    rating: u8,
}

fn lookup_customer_tool() -> Tool {
    Tool::new(
        "lookup_customer",
        json!({
            "type": "object",
            "properties": {
                "id": { "type": "string" }
            },
            "required": ["id"],
            "additionalProperties": false
        }),
    )
    .description("Look up customer details.")
}

fn build_request() -> ChatRequest {
    ChatRequest::new("demo-model")
        .system("A previous app step already gathered the review details. Extract only the final structured review.")
        .user("Customer 42 review summary: Heat deserves a 10 out of 10.")
        .tools([lookup_customer_tool()])
        .tool_choice(ToolChoice::Required)
}

fn build_provider() -> MockProvider {
    MockProvider::new([MockResponse::tool_call(
        "call_extract_1",
        "submit_structured_output",
        json!({ "title": "Heat", "rating": 10 }),
    )])
    .with_chat_capability(
        ChatCapability::StructuredOutput,
        CapabilitySupport::Unsupported,
    )
    .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported)
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let request = build_request();

    let direct = build_provider();
    let err = direct.extract::<Review>(&request).await.unwrap_err();
    match err {
        Error::Extract(inner) => match *inner {
            ExtractError::ToolConflict { mode, .. } => {
                println!(
                    "direct extract keeps caller tools, so extraction mode {mode:?} conflicts"
                );
            }
            other => {
                return Err(Error::UnexpectedResponse(format!(
                    "unexpected extract error: {other}"
                )));
            }
        },
        other => {
            return Err(Error::UnexpectedResponse(format!(
                "unexpected error: {other}"
            )));
        }
    }

    let wrapped = ExtractingProvider::new(build_provider());
    let extracted: Extracted<Review> = wrapped.extract(&request).await?;
    println!("wrapped extractor ran a dedicated extraction pass over the existing messages");
    println!("title: {}", extracted.value.title);
    println!("rating: {}", extracted.value.rating);
    println!("passes: {}", extracted.metadata.passes);

    Ok(())
}
