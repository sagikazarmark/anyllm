use anyllm::{Error, Result};
use anyllm_examples::{ALL_PROVIDER_KINDS, ProviderKind};

const VALID_SELECTION_NAMES: &[&str] = &["all", "configured", "openai", "anthropic", "gemini"];

pub fn live_selection() -> Option<String> {
    std::env::var("ANYLLM_LIVE_PROVIDER")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub fn validate_live_selection(selection: &str) -> Result<()> {
    for raw in selection.split(',') {
        let name = raw.trim();
        if name.is_empty() {
            return Err(Error::InvalidRequest(
                "ANYLLM_LIVE_PROVIDER contains an empty provider name".into(),
            ));
        }

        if !VALID_SELECTION_NAMES.contains(&name) {
            return Err(Error::InvalidRequest(format!(
                "unsupported ANYLLM_LIVE_PROVIDER entry `{name}`; expected one of: {}",
                VALID_SELECTION_NAMES.join(", ")
            )));
        }
    }

    Ok(())
}

fn configured_http_provider_kinds() -> impl Iterator<Item = ProviderKind> {
    ALL_PROVIDER_KINDS.into_iter().filter(|kind| {
        std::env::var(kind.credential_env())
            .ok()
            .is_some_and(|value| !value.trim().is_empty())
    })
}

pub fn selection_includes(selection: &str, name: &str) -> bool {
    selection
        .split(',')
        .map(str::trim)
        .any(|value| value == "all" || value == "configured" || value == name)
}

pub fn selection_uses_configured_alias(selection: &str) -> bool {
    selection
        .split(',')
        .map(str::trim)
        .any(|value| value == "all" || value == "configured")
}

pub fn ensure_selection_has_runnable_target(selection: &str) -> Result<()> {
    let has_runnable_target = if selection_uses_configured_alias(selection) {
        configured_http_provider_kinds().next().is_some()
    } else {
        selection
            .split(',')
            .map(str::trim)
            .any(|value| match value {
                "openai" => {
                    configured_http_provider_kinds().any(|kind| kind == ProviderKind::OpenAI)
                }
                "anthropic" => {
                    configured_http_provider_kinds().any(|kind| kind == ProviderKind::Anthropic)
                }
                "gemini" => {
                    configured_http_provider_kinds().any(|kind| kind == ProviderKind::Gemini)
                }
                _ => false,
            })
    };

    if has_runnable_target {
        Ok(())
    } else {
        Err(Error::InvalidRequest(format!(
            "ANYLLM_LIVE_PROVIDER={selection} selected no runnable live targets; configure at least one provider credential"
        )))
    }
}

pub fn assert_contains_token(text: &str, token: &str) {
    assert!(
        text.to_ascii_lowercase()
            .contains(&token.to_ascii_lowercase()),
        "expected response to contain '{token}', got: {text}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_live_selection_entry() {
        let err = validate_live_selection("openai,opneai").unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(message) if message.contains("opneai")));
    }

    #[test]
    fn rejects_empty_live_selection_entry() {
        let err = validate_live_selection("openai,").unwrap_err();
        assert!(
            matches!(err, Error::InvalidRequest(message) if message.contains("empty provider name"))
        );
    }

    #[test]
    fn accepts_known_live_selection_entries() {
        validate_live_selection("configured,openai").unwrap();
    }
}
