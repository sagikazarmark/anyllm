//! Parallel tool calls example: register two tools and ask a question
//! that requires both. Models that support parallel tool calls return
//! multiple `ContentBlock::ToolCall` entries in a single response.
//!
//! The crate reports `ParallelToolCalls: Supported` for OpenAI and
//! Gemini unconditionally, and `Unsupported` for Anthropic. Running
//! against Anthropic will still work but the two tool calls are issued
//! sequentially across turns rather than in one response.

use anyllm::ToolCallRef;
use anyllm::prelude::*;
use anyllm_examples::{load_provider_for_example, print_provider_banner};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct CityArg {
    city: String,
}

fn weather_tool() -> Tool {
    Tool::new(
        "lookup_weather",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
            "additionalProperties": false
        }),
    )
    .description("Look up the current weather for a city.")
}

fn time_tool() -> Tool {
    Tool::new(
        "lookup_local_time",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
            "additionalProperties": false
        }),
    )
    .description("Look up the current local time for a city.")
}

fn run_weather(call: ToolCallRef<'_>) -> Result<String> {
    let args: CityArg = call.parse_arguments()?;
    Ok(format!("Weather in {}: 18 C, clear.", args.city))
}

fn run_time(call: ToolCallRef<'_>) -> Result<String> {
    let args: CityArg = call.parse_arguments()?;
    Ok(format!("Local time in {}: 14:30.", args.city))
}

#[tokio::main]
async fn main() -> Result<()> {
    let city = std::env::args().nth(1).unwrap_or_else(|| "Paris".into());

    let target = load_provider_for_example("provider_parallel_tools", "[city]")?;
    print_provider_banner(&target);

    let mut request = ChatRequest::new(&target.model)
        .system("Use both tools when the user asks for multiple facts. Issue tool calls in parallel when possible.")
        .user(format!(
            "What is the weather AND the local time in {city}? Use the tools."
        ))
        .tools(vec![weather_tool(), time_tool()])
        .tool_choice(ToolChoice::Required)
        .parallel_tool_calls(true);

    let first = target.provider.chat(&request).await?;
    let calls: Vec<_> = first.tool_calls().collect();
    eprintln!(
        "first finish_reason={:?} tool_calls={}",
        first.finish_reason,
        calls.len()
    );

    if calls.is_empty() {
        return Err(Error::UnexpectedResponse(
            "provider returned no tool calls despite ToolChoice::Required".into(),
        ));
    }

    request.push_message(first.to_assistant_message());

    for call in &calls {
        let output = match call.name {
            "lookup_weather" => run_weather(*call)?,
            "lookup_local_time" => run_time(*call)?,
            other => {
                return Err(Error::UnexpectedResponse(format!(
                    "unknown tool call: {other}"
                )));
            }
        };
        eprintln!("tool {} -> {output}", call.name);
        request.push_tool_result(*call, output);
    }

    let second = target.provider.chat(&request).await?;
    if let Some(usage) = &second.usage {
        eprintln!("usage={usage:?}");
    }
    println!("{}", second.text_or_empty());
    Ok(())
}
