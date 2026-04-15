//! Runnable provider-backed examples for `anyllm`.

use anyllm::{DynChatProvider, Error, Result};
use anyllm_anthropic::Provider as Anthropic;
use anyllm_gemini::Provider as Gemini;
use anyllm_openai::Provider as OpenAI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    Gemini,
    OpenAI,
}

pub const ALL_PROVIDER_KINDS: [ProviderKind; 3] = [
    ProviderKind::Anthropic,
    ProviderKind::Gemini,
    ProviderKind::OpenAI,
];

fn configured_provider_kinds() -> Vec<ProviderKind> {
    ALL_PROVIDER_KINDS
        .into_iter()
        .filter(|kind| {
            std::env::var(kind.credential_env())
                .ok()
                .is_some_and(|value| !value.trim().is_empty())
        })
        .collect()
}

fn provider_env_summary(kind: ProviderKind) -> String {
    let optional_envs = if kind.optional_envs().is_empty() {
        "none".to_string()
    } else {
        kind.optional_envs().join(", ")
    };

    format!(
        "{} => credential: {}, optional: {}, model override: {}",
        kind.as_str(),
        kind.credential_env(),
        optional_envs,
        kind.model_env()
    )
}

fn usage_help(example_name: &str, args: &str) -> String {
    let provider_help = ALL_PROVIDER_KINDS
        .into_iter()
        .map(provider_env_summary)
        .collect::<Vec<_>>()
        .join("; ");

    format!(
        "Usage: PROVIDER=<openai|anthropic|gemini> cargo run -p anyllm-examples --example {example_name} -- {args}\nProvider envs: {provider_help}\nIf PROVIDER is unset, the loader auto-selects only when exactly one provider credential is configured."
    )
}

pub struct LoadedProvider {
    pub provider: DynChatProvider,
    pub kind: ProviderKind,
    pub model: String,
    pub credential_env: &'static str,
    pub optional_envs: &'static [&'static str],
    pub model_env: &'static str,
}

impl ProviderKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
            Self::OpenAI => "openai",
        }
    }

    #[must_use]
    pub fn default_model(self) -> &'static str {
        match self {
            Self::Anthropic => "claude-sonnet-4-20250514",
            Self::Gemini => "gemini-2.5-pro",
            Self::OpenAI => "gpt-4o",
        }
    }

    #[must_use]
    pub fn credential_env(self) -> &'static str {
        match self {
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::Gemini => "GEMINI_API_KEY",
            Self::OpenAI => "OPENAI_API_KEY",
        }
    }

    #[must_use]
    pub fn optional_envs(self) -> &'static [&'static str] {
        match self {
            Self::Anthropic => &["ANTHROPIC_BASE_URL"],
            Self::Gemini => &["GEMINI_BASE_URL"],
            Self::OpenAI => &["OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"],
        }
    }

    #[must_use]
    pub fn model_env(self) -> &'static str {
        match self {
            Self::Anthropic => "ANTHROPIC_MODEL",
            Self::Gemini => "GEMINI_MODEL",
            Self::OpenAI => "OPENAI_MODEL",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "anthropic" => Ok(Self::Anthropic),
            "gemini" => Ok(Self::Gemini),
            "openai" => Ok(Self::OpenAI),
            other => Err(Error::InvalidRequest(format!(
                "unsupported PROVIDER={other}; expected one of: openai, anthropic, gemini"
            ))),
        }
    }
}

#[must_use]
pub fn usage(example_name: &str, args: &str) -> String {
    usage_help(example_name, args)
}

pub fn load_provider(kind: ProviderKind) -> Result<LoadedProvider> {
    let model = std::env::var(kind.model_env())
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| kind.default_model().to_string());

    let provider = match kind {
        ProviderKind::Anthropic => DynChatProvider::new(Anthropic::from_env()?),
        ProviderKind::Gemini => DynChatProvider::new(Gemini::from_env()?),
        ProviderKind::OpenAI => DynChatProvider::new(OpenAI::from_env()?),
    };

    Ok(LoadedProvider {
        provider,
        kind,
        model,
        credential_env: kind.credential_env(),
        optional_envs: kind.optional_envs(),
        model_env: kind.model_env(),
    })
}

fn missing_provider_selection_error() -> Error {
    Error::InvalidRequest("PROVIDER is not set and no provider credentials were found".into())
}

fn ambiguous_provider_selection_error(kinds: &[ProviderKind]) -> Error {
    Error::InvalidRequest(format!(
        "PROVIDER is not set and multiple providers are configured ({}). Set PROVIDER explicitly",
        kinds
            .iter()
            .map(|kind| kind.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

pub fn load_provider_from_env() -> Result<LoadedProvider> {
    if let Ok(provider) = std::env::var("PROVIDER") {
        return load_provider(ProviderKind::parse(provider.trim())?);
    }

    let configured = configured_provider_kinds();
    match configured.as_slice() {
        [kind] => load_provider(*kind),
        [] => Err(missing_provider_selection_error()),
        kinds => Err(ambiguous_provider_selection_error(kinds)),
    }
}

pub fn load_provider_for_example(example_name: &str, args: &str) -> Result<LoadedProvider> {
    load_provider_from_env().map_err(|err| match err {
        Error::InvalidRequest(message) => {
            Error::InvalidRequest(format!("{message}. {}", usage_help(example_name, args)))
        }
        other => other,
    })
}

pub fn print_provider_banner(target: &LoadedProvider) {
    eprintln!(
        "using provider={} model={}",
        target.kind.as_str(),
        target.model
    );
    eprintln!("required env: {}", target.credential_env);
    if !target.optional_envs.is_empty() {
        eprintln!("optional envs: {}", target.optional_envs.join(", "));
    }
    eprintln!("model override env: {}", target.model_env);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{LazyLock, Mutex};

    static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    struct EnvVarGuard {
        name: &'static str,
        original: Option<String>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: Option<&str>) -> Self {
            let original = std::env::var(name).ok();

            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }

            Self { name, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.original {
                    Some(value) => std::env::set_var(self.name, value),
                    None => std::env::remove_var(self.name),
                }
            }
        }
    }

    #[test]
    fn usage_lists_provider_envs() {
        let text = usage("provider_chat", "[prompt]");
        assert!(text.contains("OPENAI_API_KEY"));
        assert!(text.contains("ANTHROPIC_API_KEY"));
        assert!(text.contains("GEMINI_API_KEY"));
        assert!(
            text.contains("auto-selects only when exactly one provider credential is configured")
        );
    }

    #[test]
    fn load_provider_from_env_autoselects_single_configured_provider() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _provider = EnvVarGuard::set("PROVIDER", None);
        let _openai = EnvVarGuard::set("OPENAI_API_KEY", None);
        let _anthropic = EnvVarGuard::set("ANTHROPIC_API_KEY", None);
        let _gemini = EnvVarGuard::set("GEMINI_API_KEY", Some("gemini-test-key"));
        let _model = EnvVarGuard::set("GEMINI_MODEL", None);

        let loaded = load_provider_from_env().unwrap();
        assert_eq!(loaded.kind, ProviderKind::Gemini);
        assert_eq!(loaded.model, ProviderKind::Gemini.default_model());
    }

    #[test]
    fn load_provider_from_env_rejects_when_no_provider_is_configured() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _provider = EnvVarGuard::set("PROVIDER", None);
        let _openai = EnvVarGuard::set("OPENAI_API_KEY", None);
        let _anthropic = EnvVarGuard::set("ANTHROPIC_API_KEY", None);
        let _gemini = EnvVarGuard::set("GEMINI_API_KEY", None);

        match load_provider_from_env() {
            Ok(_) => panic!("expected missing provider configuration to be rejected"),
            Err(Error::InvalidRequest(message)) => {
                assert!(
                    message.contains("PROVIDER is not set and no provider credentials were found")
                );
                assert!(!message.contains("OPENAI_API_KEY"));
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn load_provider_from_env_rejects_ambiguous_selection() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _provider = EnvVarGuard::set("PROVIDER", None);
        let _openai = EnvVarGuard::set("OPENAI_API_KEY", Some("openai-test-key"));
        let _anthropic = EnvVarGuard::set("ANTHROPIC_API_KEY", Some("anthropic-test-key"));
        let _gemini = EnvVarGuard::set("GEMINI_API_KEY", None);

        match load_provider_from_env() {
            Ok(_) => panic!("expected ambiguous provider configuration to be rejected"),
            Err(Error::InvalidRequest(message)) => {
                assert!(message.contains("multiple providers are configured"));
                assert!(message.contains("openai"));
                assert!(message.contains("anthropic"));
                assert!(message.contains("Set PROVIDER explicitly"));
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn load_provider_for_example_appends_example_specific_usage() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _provider = EnvVarGuard::set("PROVIDER", None);
        let _openai = EnvVarGuard::set("OPENAI_API_KEY", None);
        let _anthropic = EnvVarGuard::set("ANTHROPIC_API_KEY", None);
        let _gemini = EnvVarGuard::set("GEMINI_API_KEY", None);

        match load_provider_for_example("provider_tools", "[city]") {
            Ok(_) => panic!("expected missing provider configuration to be rejected"),
            Err(Error::InvalidRequest(message)) => {
                assert!(message.contains("provider_tools"));
                assert!(message.contains("OPENAI_API_KEY"));
                assert!(message.contains("ANTHROPIC_API_KEY"));
                assert!(message.contains("GEMINI_API_KEY"));
                assert!(!message.contains("provider_chat"));
            }
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
        }
    }
}
