//! Pre-configured provider factories for common OpenAI-compatible APIs.
//!
//! Each provider type is a zero-sized struct with factory methods that return
//! a [`Provider`](crate::Provider) configured with the correct
//! base URL, auth style, and capabilities.
//!
//! # Example
//!
//! ```rust,no_run
//! use anyllm::prelude::*;
//! use anyllm_openai_compat::providers::Cloudflare;
//!
//! # async fn example() -> anyllm::Result<()> {
//! let provider = Cloudflare::from_env()?;
//! let response = provider.chat(
//!     &ChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
//!         .message(Message::user("Hello!"))
//! ).await?;
//! # Ok(())
//! # }
//! ```

mod cloudflare;

pub use cloudflare::Cloudflare;
