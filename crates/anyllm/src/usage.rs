use std::ops::{Add, AddAssign, Sub};

use serde::{Deserialize, Serialize};

/// Token usage statistics for a chat completion.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Prompt/input tokens billed for the request.
    pub input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Completion/output tokens billed for the response.
    pub output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Total billed tokens, when the provider reports it directly.
    pub total_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Input tokens served from cache.
    pub cached_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Tokens spent creating a cache entry.
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Hidden reasoning tokens, when separately reported.
    pub reasoning_tokens: Option<u64>,
}

impl Usage {
    /// Create an empty usage value.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the prompt/input token count.
    #[must_use]
    pub fn input_tokens(mut self, input_tokens: u64) -> Self {
        self.input_tokens = Some(input_tokens);
        self
    }

    /// Set the completion/output token count.
    #[must_use]
    pub fn output_tokens(mut self, output_tokens: u64) -> Self {
        self.output_tokens = Some(output_tokens);
        self
    }

    /// Set the total token count.
    #[must_use]
    pub fn total_tokens(mut self, total_tokens: u64) -> Self {
        self.total_tokens = Some(total_tokens);
        self
    }

    /// Set the cached input token count.
    #[must_use]
    pub fn cached_input_tokens(mut self, cached_input_tokens: u64) -> Self {
        self.cached_input_tokens = Some(cached_input_tokens);
        self
    }

    /// Set the cache-creation input token count.
    #[must_use]
    pub fn cache_creation_input_tokens(mut self, cache_creation_input_tokens: u64) -> Self {
        self.cache_creation_input_tokens = Some(cache_creation_input_tokens);
        self
    }

    /// Set the reasoning token count.
    #[must_use]
    pub fn reasoning_tokens(mut self, reasoning_tokens: u64) -> Self {
        self.reasoning_tokens = Some(reasoning_tokens);
        self
    }

    /// Returns the total token count.
    ///
    /// Prefers `total_tokens` if set, otherwise sums `input_tokens + output_tokens`
    /// (only if both are `Some`). Returns `None` if neither approach yields a value.
    pub fn total(&self) -> Option<u64> {
        if self.total_tokens.is_some() {
            return self.total_tokens;
        }
        match (self.input_tokens, self.output_tokens) {
            (Some(i), Some(o)) => Some(i + o),
            _ => None,
        }
    }
}

/// Merges two `Usage` values field-by-field.
///
/// For each `Option<u64>` field:
/// - `None + None = None`
/// - `None + Some(n) = Some(n)`
/// - `Some(a) + Some(b) = Some(a + b)`
impl Add for Usage {
    type Output = Usage;

    fn add(self, rhs: Usage) -> Usage {
        Usage {
            input_tokens: add_options(self.input_tokens, rhs.input_tokens),
            output_tokens: add_options(self.output_tokens, rhs.output_tokens),
            total_tokens: add_options(self.total_tokens, rhs.total_tokens),
            cached_input_tokens: add_options(self.cached_input_tokens, rhs.cached_input_tokens),
            cache_creation_input_tokens: add_options(
                self.cache_creation_input_tokens,
                rhs.cache_creation_input_tokens,
            ),
            reasoning_tokens: add_options(self.reasoning_tokens, rhs.reasoning_tokens),
        }
    }
}

impl Add<&Usage> for Usage {
    type Output = Usage;

    fn add(self, rhs: &Usage) -> Usage {
        Usage {
            input_tokens: add_options(self.input_tokens, rhs.input_tokens),
            output_tokens: add_options(self.output_tokens, rhs.output_tokens),
            total_tokens: add_options(self.total_tokens, rhs.total_tokens),
            cached_input_tokens: add_options(self.cached_input_tokens, rhs.cached_input_tokens),
            cache_creation_input_tokens: add_options(
                self.cache_creation_input_tokens,
                rhs.cache_creation_input_tokens,
            ),
            reasoning_tokens: add_options(self.reasoning_tokens, rhs.reasoning_tokens),
        }
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, rhs: Usage) {
        *self += &rhs;
    }
}

impl AddAssign<&Usage> for Usage {
    fn add_assign(&mut self, rhs: &Usage) {
        self.input_tokens = add_options(self.input_tokens, rhs.input_tokens);
        self.output_tokens = add_options(self.output_tokens, rhs.output_tokens);
        self.total_tokens = add_options(self.total_tokens, rhs.total_tokens);
        self.cached_input_tokens = add_options(self.cached_input_tokens, rhs.cached_input_tokens);
        self.cache_creation_input_tokens = add_options(
            self.cache_creation_input_tokens,
            rhs.cache_creation_input_tokens,
        );
        self.reasoning_tokens = add_options(self.reasoning_tokens, rhs.reasoning_tokens);
    }
}

/// Subtracts `rhs` from `self` field-by-field (saturating).
///
/// For each `Option<u64>` field:
/// - `Some(a) - Some(b) = Some(a.saturating_sub(b))`
/// - `Some(a) - None = Some(a)`
/// - `None - Some(_) = None`
/// - `None - None = None`
impl Sub for Usage {
    type Output = Usage;

    fn sub(self, rhs: Usage) -> Usage {
        self - &rhs
    }
}

impl Sub<&Usage> for Usage {
    type Output = Usage;

    fn sub(self, rhs: &Usage) -> Usage {
        Usage {
            input_tokens: sub_options(self.input_tokens, rhs.input_tokens),
            output_tokens: sub_options(self.output_tokens, rhs.output_tokens),
            total_tokens: sub_options(self.total_tokens, rhs.total_tokens),
            cached_input_tokens: sub_options(self.cached_input_tokens, rhs.cached_input_tokens),
            cache_creation_input_tokens: sub_options(
                self.cache_creation_input_tokens,
                rhs.cache_creation_input_tokens,
            ),
            reasoning_tokens: sub_options(self.reasoning_tokens, rhs.reasoning_tokens),
        }
    }
}

fn sub_options(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.saturating_sub(b)),
        (Some(a), None) => Some(a),
        (None, _) => None,
    }
}

fn add_options(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.saturating_add(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_prefers_total_tokens_when_set() {
        let u = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            total_tokens: Some(42),
            ..Default::default()
        };
        assert_eq!(u.total(), Some(42));
    }

    #[test]
    fn total_falls_back_to_input_plus_output() {
        let u = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            ..Default::default()
        };
        assert_eq!(u.total(), Some(30));
    }

    #[test]
    fn total_returns_none_without_complete_token_totals() {
        let cases = [
            Usage {
                input_tokens: Some(10),
                ..Default::default()
            },
            Usage {
                output_tokens: Some(20),
                ..Default::default()
            },
            Usage::default(),
        ];

        for usage in cases {
            assert_eq!(usage.total(), None);
        }
    }

    #[test]
    fn add_none_plus_none_is_none() {
        let a = Usage::default();
        let b = Usage::default();
        let result = a + b;
        assert_eq!(result, Usage::default());
    }

    #[test]
    fn add_none_plus_some_is_some() {
        let a = Usage::default();
        let b = Usage {
            input_tokens: Some(5),
            output_tokens: Some(10),
            total_tokens: Some(15),
            cached_input_tokens: Some(3),
            cache_creation_input_tokens: Some(2),
            reasoning_tokens: Some(1),
        };
        let result = a + b.clone();
        assert_eq!(result, b);
    }

    #[test]
    fn add_some_plus_some_sums() {
        let a = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            total_tokens: Some(30),
            cached_input_tokens: Some(5),
            cache_creation_input_tokens: Some(3),
            reasoning_tokens: Some(2),
        };
        let b = Usage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            total_tokens: Some(3),
            cached_input_tokens: Some(1),
            cache_creation_input_tokens: Some(1),
            reasoning_tokens: Some(1),
        };
        let result = a + b;
        assert_eq!(
            result,
            Usage {
                input_tokens: Some(11),
                output_tokens: Some(22),
                total_tokens: Some(33),
                cached_input_tokens: Some(6),
                cache_creation_input_tokens: Some(4),
                reasoning_tokens: Some(3),
            }
        );
    }

    #[test]
    fn add_some_plus_none_is_some() {
        let a = Usage {
            input_tokens: Some(10),
            ..Default::default()
        };
        let b = Usage::default();
        let result = a + b;
        assert_eq!(result.input_tokens, Some(10));
        assert_eq!(result.output_tokens, None);
    }

    #[test]
    fn add_assign_works_same_as_add() {
        let a = Usage {
            input_tokens: Some(10),
            output_tokens: Some(20),
            ..Default::default()
        };
        let b = Usage {
            input_tokens: Some(5),
            output_tokens: None,
            total_tokens: Some(7),
            cached_input_tokens: Some(3),
            ..Default::default()
        };
        let expected = a.clone() + b.clone();
        let mut a_mut = a;
        a_mut += b;
        assert_eq!(a_mut, expected);
    }

    #[test]
    fn add_assign_none_plus_some() {
        let mut a = Usage::default();
        let b = Usage {
            input_tokens: Some(42),
            ..Default::default()
        };
        a += b;
        assert_eq!(a.input_tokens, Some(42));
        assert_eq!(a.output_tokens, None);
    }

    #[test]
    fn serde_round_trip_handles_full_and_sparse_usage() {
        let cases = [
            Usage {
                input_tokens: Some(100),
                output_tokens: Some(200),
                total_tokens: Some(300),
                cached_input_tokens: Some(50),
                cache_creation_input_tokens: Some(25),
                reasoning_tokens: Some(10),
            },
            Usage {
                input_tokens: Some(100),
                ..Default::default()
            },
            Usage::default(),
        ];

        for usage in cases {
            let json = serde_json::to_string(&usage).unwrap();
            let deserialized: Usage = serde_json::from_str(&json).unwrap();
            assert_eq!(usage, deserialized);
        }
    }
}
