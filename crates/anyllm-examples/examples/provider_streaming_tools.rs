//! Streaming + tool calls example: iterate raw [`StreamEvent`]s and
//! watch tool-call blocks arrive mid-stream, then reconstruct the full
//! [`ChatResponse`] via [`StreamCollector`].
//!
//! Works against any configured provider. Tool-call arguments arrive as
//! incremental `ToolCallDelta` events whose `arguments` field carries a
//! JSON string that accumulates across deltas.

use anyllm::prelude::*;
use anyllm_examples::{load_provider_for_example, print_provider_banner};
use serde_json::json;

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

#[tokio::main]
async fn main() -> Result<()> {
    let city = std::env::args().nth(1).unwrap_or_else(|| "Berlin".into());

    let target = load_provider_for_example("provider_streaming_tools", "[city]")?;
    print_provider_banner(&target);

    let request = ChatRequest::new(&target.model)
        .system("Use the weather tool when asked about a city's weather.")
        .user(format!("What is the weather in {city}?"))
        .tools(vec![weather_tool()])
        .tool_choice(ToolChoice::Required);

    let mut stream = target.provider.chat_stream(&request).await?;
    let mut collector = StreamCollector::new();

    while let Some(event) = stream.next().await {
        let event = event?;
        match &event {
            StreamEvent::BlockStart {
                block_type: StreamBlockType::ToolCall,
                id,
                name,
                ..
            } => {
                eprintln!(
                    "tool block start: name={} id={}",
                    name.as_deref().unwrap_or("?"),
                    id.as_deref().unwrap_or("?"),
                );
            }
            StreamEvent::ToolCallDelta { arguments, .. } => {
                eprint!("{arguments}");
            }
            StreamEvent::BlockStop { .. } => {
                eprintln!();
            }
            StreamEvent::TextDelta { text, .. } => {
                print!("{text}");
            }
            _ => {}
        }
        collector.push(event)?;
    }
    println!();

    let response = collector.finish()?;
    eprintln!("finish_reason={:?}", response.finish_reason);

    for call in response.tool_calls() {
        eprintln!(
            "collected tool call: name={} id={} args={}",
            call.name, call.id, call.arguments
        );
    }

    Ok(())
}
