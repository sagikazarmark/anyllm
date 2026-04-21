use anyllm::ToolCallRef;
use anyllm::prelude::*;
use anyllm_examples::{load_provider_for_example, print_provider_banner};
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

fn run_weather_tool(call: ToolCallRef<'_>) -> Result<String> {
    let args: WeatherArgs = call.parse_arguments()?;
    let city = args.city.trim();

    let forecast = match city.to_ascii_lowercase().as_str() {
        "san francisco" => "cool and foggy, 58 F",
        "new york" => "bright and windy, 67 F",
        "london" => "overcast with light rain, 54 F",
        _ => "mild with scattered clouds, 72 F",
    };

    Ok(format!("Weather for {city}: {forecast}."))
}

#[tokio::main]
async fn main() -> Result<()> {
    let city = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "San Francisco".into());

    let target = load_provider_for_example("provider_tools", "[city]")?;
    print_provider_banner(&target);

    let mut request = ChatRequest::new(&target.model)
        .system("You are a concise assistant. Use the weather tool when a city lookup is needed.")
        .user(format!(
            "What is the weather in {city}? Answer in one sentence."
        ))
        .tools(vec![weather_tool()])
        .tool_choice(ToolChoice::Required);

    let first = target.provider.chat(&request).await?;
    eprintln!("first_finish_reason={:?}", first.finish_reason);

    if !first.has_tool_calls() {
        return Err(Error::UnexpectedResponse(
            "provider returned no tool call despite ToolChoice::Required".into(),
        ));
    }

    request.push_message(first.to_assistant_message());

    for call in first.tool_calls() {
        eprintln!("tool_call name={} id={}", call.name, call.id);
        let tool_output = run_weather_tool(call)?;
        eprintln!("tool_result {tool_output}");
        request.push_tool_result(call, tool_output);
    }

    let second = target.provider.chat(&request).await?;

    eprintln!("final_finish_reason={:?}", second.finish_reason);
    if let Some(usage) = &second.usage {
        eprintln!("usage={usage:?}");
    }
    println!("{}", second.text_or_empty());
    Ok(())
}
