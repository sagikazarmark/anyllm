mod support;

use anyllm::{Error, Result};

const PROVIDER_NAME: &str = "cloudflare-worker";
const SMOKE_URL_ENV: &str = "ANYLLM_CLOUDFLARE_WORKER_SMOKE_URL";
const DEFAULT_MODEL_ENV: &str = "CLOUDFLARE_WORKER_MODEL";
const TOOL_MODEL_ENV: &str = "CLOUDFLARE_WORKER_TOOL_MODEL";
const JSON_MODEL_ENV: &str = "CLOUDFLARE_WORKER_JSON_MODEL";

fn smoke_url_is_set() -> bool {
    std::env::var(SMOKE_URL_ENV)
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

fn required_env_var(name: &'static str) -> Result<String> {
    let value =
        std::env::var(name).map_err(|_| Error::InvalidRequest(format!("{name} not set")))?;
    if value.trim().is_empty() {
        return Err(Error::InvalidRequest(format!("{name} not set")));
    }
    Ok(value)
}

fn selected_model(model_env: &'static str) -> Option<String> {
    std::env::var(model_env)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn smoke_target(selection: &str) -> Result<Option<String>> {
    if !support::selection_includes(selection, PROVIDER_NAME) {
        return Ok(None);
    }

    if !smoke_url_is_set() {
        if support::selection_uses_configured_alias(selection) {
            eprintln!("skipping {PROVIDER_NAME} live checks; {SMOKE_URL_ENV} is not set");
            return Ok(None);
        }

        return Err(Error::InvalidRequest(format!(
            "{PROVIDER_NAME} selected by ANYLLM_LIVE_PROVIDER but {SMOKE_URL_ENV} is not set"
        )));
    }

    Ok(Some(required_env_var(SMOKE_URL_ENV)?))
}

fn smoke_endpoint(
    base_url: &str,
    path: &str,
    token: &str,
    model_env: &'static str,
) -> Result<reqwest::Url> {
    let normalized = if base_url.ends_with('/') {
        base_url.to_string()
    } else {
        format!("{base_url}/")
    };

    let mut url = reqwest::Url::parse(&normalized)
        .map_err(|err| Error::InvalidRequest(format!("invalid {SMOKE_URL_ENV}: {err}")))?
        .join(path)
        .map_err(|err| Error::InvalidRequest(format!("invalid {SMOKE_URL_ENV}: {err}")))?;

    {
        let mut pairs = url.query_pairs_mut();
        if path == "chat" || path == "stream" {
            pairs.append_pair(
                "prompt",
                &format!("Reply with exactly {token} and nothing else."),
            );
        } else {
            pairs.append_pair("token", token);
        }
        if let Some(model) = selected_model(model_env).or_else(|| selected_model(DEFAULT_MODEL_ENV))
        {
            pairs.append_pair("model", &model);
        }
    }

    Ok(url)
}

async fn run_cloudflare_worker_smoke(
    base_url: &str,
    path: &str,
    token: &str,
    model_env: &'static str,
) -> Result<()> {
    let url = smoke_endpoint(base_url, path, token, model_env)?;
    eprintln!("using provider={PROVIDER_NAME} url={url}");
    if let Some(model) = selected_model(model_env) {
        eprintln!("model override env: {model_env}={model}");
    } else if let Some(model) = selected_model(DEFAULT_MODEL_ENV) {
        eprintln!("model override env: {DEFAULT_MODEL_ENV}={model}");
    }

    let response = reqwest::Client::new()
        .get(url)
        .send()
        .await
        .map_err(|err| Error::Provider {
            status: None,
            message: err.to_string(),
            body: None,
            request_id: None,
        })?;

    let response = response.error_for_status().map_err(|err| Error::Provider {
        status: err.status().map(|status| status.as_u16()),
        message: err.to_string(),
        body: None,
        request_id: None,
    })?;

    let text = response.text().await.map_err(|err| Error::Provider {
        status: None,
        message: err.to_string(),
        body: None,
        request_id: None,
    })?;

    support::assert_contains_token(&text, token);
    Ok(())
}

#[tokio::test]
async fn cloudflare_worker_live_smoke() -> Result<()> {
    let Some(selection) = support::live_selection() else {
        eprintln!(
            "skipping {PROVIDER_NAME} live checks; set ANYLLM_LIVE_PROVIDER={PROVIDER_NAME} or all"
        );
        return Ok(());
    };

    support::validate_live_selection(&selection)?;

    if !support::selection_includes(&selection, PROVIDER_NAME) {
        eprintln!("skipping {PROVIDER_NAME} live checks; ANYLLM_LIVE_PROVIDER={selection}");
        return Ok(());
    }

    support::ensure_selection_has_runnable_target(&selection)?;

    let Some(base_url) = smoke_target(&selection)? else {
        return Ok(());
    };

    run_cloudflare_worker_smoke(&base_url, "chat", "cf-worker-chat-ok", DEFAULT_MODEL_ENV).await?;
    run_cloudflare_worker_smoke(
        &base_url,
        "stream",
        "cf-worker-stream-ok",
        DEFAULT_MODEL_ENV,
    )
    .await?;
    run_cloudflare_worker_smoke(&base_url, "tools", "cf-worker-tool-ok", TOOL_MODEL_ENV).await?;
    run_cloudflare_worker_smoke(&base_url, "json", "cf-worker-json-ok", JSON_MODEL_ENV).await?;
    Ok(())
}
