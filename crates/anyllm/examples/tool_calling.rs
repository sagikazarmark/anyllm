use anyllm::ToolCallRef;
use anyllm::prelude::*;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct WeatherArgs {
    city: String,
}

fn weather_tool() -> Tool {
    Tool::new(
        "lookup_weather",
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name to look up"
                }
            },
            "required": ["city"],
            "additionalProperties": false
        }),
    )
    .description("Look up a fake weather report for a city.")
}

fn build_provider() -> MockProvider {
    MockProvider::tool_round_trip(
        "call_weather_1",
        "lookup_weather",
        json!({ "city": "San Francisco" }),
        "The weather in San Francisco is cool and foggy.",
    )
}

fn build_request() -> ChatRequest {
    ChatRequest::new("demo-model")
        .system("You are a concise assistant. Use the weather tool when a city lookup is needed.")
        .user("What is the weather in San Francisco?")
        .tools([weather_tool()])
        .tool_choice(ToolChoice::Required)
}

fn run_weather_tool(call: ToolCallRef<'_>) -> anyllm::Result<String> {
    let args: WeatherArgs = call.parse_arguments()?;
    Ok(format!("Weather for {}: cool and foggy.", args.city))
}

#[tokio::main]
async fn main() -> anyllm::Result<()> {
    let provider = build_provider();
    let mut request = build_request();

    let first_turn = provider.chat(&request).await?;
    let tool_calls: Vec<_> = first_turn.tool_calls().collect();
    println!("tool calls: {}", tool_calls.len());

    request.push_message(first_turn.to_assistant_message());

    for call in tool_calls {
        let tool_output = run_weather_tool(call)?;
        println!("tool call name={} id={}", call.name, call.id);
        request.push_tool_result(call, tool_output);
    }

    let second_turn = provider.chat(&request).await?;
    println!("final text: {}", second_turn.text_or_empty());
    println!("calls recorded: {}", provider.call_count());

    Ok(())
}
