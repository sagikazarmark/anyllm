use std::fs;
use std::path::PathBuf;

use anyllm::prelude::*;
use anyllm::{ChatRequestRecord, ChatResponseRecord};
use anyllm_examples::{load_provider_from_env, print_provider_banner, usage};

fn artifact_paths(provider_name: &str) -> (PathBuf, PathBuf) {
    let base = std::env::temp_dir().join(format!(
        "anyllm-record-replay-{}-{}",
        provider_name,
        std::process::id()
    ));
    (
        base.with_extension("request.json"),
        base.with_extension("response.json"),
    )
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| {
        "Summarize why portable request/response records are useful for debugging despite lossy replay.".into()
    });

    let target = load_provider_from_env().map_err(|err| {
        anyllm::Error::InvalidRequest(format!("{err}. {}", usage("record_replay", "[prompt]")))
    })?;
    let (request_path, response_path) = artifact_paths(target.kind.as_str());

    print_provider_banner(&target);

    let request = ChatRequest::new(&target.model)
        .system("You are a concise assistant.")
        .user(prompt)
        .temperature(0.2)
        .max_tokens(120);

    let request_record = ChatRequestRecord::from(&request);
    let request_json =
        serde_json::to_string_pretty(&request_record).map_err(anyllm::Error::from)?;
    fs::write(&request_path, request_json)?;
    eprintln!("wrote request record to {}", request_path.display());

    let replay_json = fs::read_to_string(&request_path)?;
    let replay_record: ChatRequestRecord =
        serde_json::from_str(&replay_json).map_err(anyllm::Error::from)?;
    let replay_request = replay_record.into_chat_request_lossy();

    let response = target.provider.chat(&replay_request).await?;
    let response_record = ChatResponseRecord::from(&response);
    let response_json =
        serde_json::to_string_pretty(&response_record).map_err(anyllm::Error::from)?;
    fs::write(&response_path, response_json)?;
    eprintln!("wrote response record to {}", response_path.display());

    let replayed_response_json = fs::read_to_string(&response_path)?;
    let replayed_response_record: ChatResponseRecord =
        serde_json::from_str(&replayed_response_json).map_err(anyllm::Error::from)?;
    let replayed_response = replayed_response_record.into_chat_response_lossy();
    eprintln!(
        "replayed response text: {}",
        replayed_response.text_or_empty()
    );

    eprintln!(
        "note: replay uses lossy rebuilds; typed RequestOptions are dropped, and typed ResponseMetadata comes back as portable JSON only"
    );
    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }

    println!("{}", response.text_or_empty());
    Ok(())
}
