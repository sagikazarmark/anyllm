//! Reasoning example: enable thinking/reasoning on the target provider
//! and print the reasoning text alongside the final answer.
//!
//! Reasoning support is model-dependent:
//!
//! - Anthropic: the default model (claude-sonnet-4-*) supports extended
//!   thinking out of the box.
//! - Gemini: the default model (gemini-2.5-pro) returns reasoning blocks.
//! - OpenAI: the default `gpt-4o` does NOT accept `reasoning_effort`.
//!   Set `OPENAI_MODEL` to `o3-mini`, `o4-mini`, or `gpt-5` to exercise
//!   this example against OpenAI.

use anyllm::prelude::*;
use anyllm_examples::{load_provider_for_example, print_provider_banner};

#[tokio::main]
async fn main() -> Result<()> {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| {
        "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning.".into()
    });

    let target = load_provider_for_example("provider_reasoning", "[prompt]")?;
    print_provider_banner(&target);

    let request = ChatRequest::new(&target.model)
        .reasoning(ReasoningConfig {
            enabled: true,
            budget_tokens: None,
            effort: Some(ReasoningEffort::High),
        })
        .user(prompt);

    let response = target.provider.chat(&request).await?;

    if let Some(finish_reason) = &response.finish_reason {
        eprintln!("finish_reason={finish_reason:?}");
    }
    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }

    if let Some(reasoning) = response.reasoning_text() {
        println!("--- reasoning ---");
        println!("{reasoning}");
        println!("--- answer ---");
    } else {
        eprintln!("(no reasoning blocks surfaced; provider may not expose reasoning content)");
    }
    println!("{}", response.text_or_empty());
    Ok(())
}
