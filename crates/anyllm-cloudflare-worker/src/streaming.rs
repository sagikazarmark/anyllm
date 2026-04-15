//! SSE parsing for Cloudflare Workers AI streaming responses.
//!
//! When `stream: true` is set, `Ai::run_bytes()` returns a `ByteStream`
//! yielding raw SSE bytes. Each event looks like:
//!
//! ```text
//! data: {"response":"token_text"}
//!
//! data: [DONE]
//! ```
//!
//! This module parses those bytes into anyllm `StreamEvent`s.
//!
//! # Send safety
//!
//! `worker::ByteStream` is `!Send` because it wraps JS objects via wasm-bindgen.
//! However, Cloudflare Workers run on a single thread, so `Send` is irrelevant
//! at runtime. We use `send_wrapper::SendWrapper` to satisfy the `Send` bound
//! required by `ChatStream = Pin<Box<dyn Stream + Send>>`.

use std::collections::VecDeque;
use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll};

use anyllm::{
    ChatStream, Error, ExtraMap, Result, StreamBlockType, StreamEvent, UsageMetadataMode,
};
use futures_util::Stream;
use send_wrapper::SendWrapper;

use crate::wire;

/// State machine for parsing Cloudflare SSE events into anyllm StreamEvents.
struct SseState {
    started: bool,
    finished: bool,
    text_block_open: bool,
    buffer: Vec<u8>,
}

impl SseState {
    fn new() -> Self {
        Self {
            started: false,
            finished: false,
            text_block_open: false,
            buffer: Vec::new(),
        }
    }

    fn process_bytes(&mut self, bytes: &[u8]) -> Vec<Result<StreamEvent>> {
        self.buffer.extend_from_slice(bytes);

        let mut events = Vec::new();

        // Process complete lines
        while let Some(newline_pos) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line_bytes: Vec<u8> = self.buffer.drain(..=newline_pos).collect();
            line_bytes.pop();
            if line_bytes.ends_with(b"\r") {
                line_bytes.pop();
            }

            if line_bytes.is_empty() {
                continue;
            }

            let line = match std::str::from_utf8(&line_bytes) {
                Ok(line) => line,
                Err(e) => {
                    return vec![Err(Error::Stream(format!(
                        "Invalid UTF-8 in SSE stream: {e}"
                    )))];
                }
            };

            let line_events = self.process_line(line);
            events.extend(line_events);
        }

        events
    }

    fn process_line(&mut self, line: &str) -> Vec<Result<StreamEvent>> {
        if self.finished {
            return Vec::new();
        }

        let data = if let Some(stripped) = line.strip_prefix("data: ") {
            stripped
        } else if let Some(stripped) = line.strip_prefix("data:") {
            stripped
        } else {
            return Vec::new();
        };

        if data.is_empty() {
            return Vec::new();
        }

        if data == "[DONE]" {
            self.finished = true;
            return self.finalize_complete();
        }

        let chunk: wire::StreamChunk = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                return vec![Err(Error::Stream(format!("SSE parse error: {e}")))];
            }
        };

        let mut events = Vec::new();

        if !self.started {
            self.started = true;
            events.push(Ok(StreamEvent::ResponseStart {
                id: None,
                model: None,
            }));
        }

        if let Some(text) = chunk.response
            && !text.is_empty()
        {
            if !self.text_block_open {
                self.text_block_open = true;
                events.push(Ok(StreamEvent::BlockStart {
                    index: 0,
                    block_type: StreamBlockType::Text,
                    id: None,
                    name: None,
                    type_name: None,
                    data: None,
                }));
            }
            events.push(Ok(StreamEvent::TextDelta { index: 0, text }));
        }

        events
    }

    fn drain_trailing_buffered_line(&mut self) -> Vec<Result<StreamEvent>> {
        if self.buffer.is_empty() {
            return Vec::new();
        }

        let mut line_bytes = std::mem::take(&mut self.buffer);
        if line_bytes.ends_with(b"\r") {
            line_bytes.pop();
        }

        if line_bytes.is_empty() {
            return Vec::new();
        }

        let line = match std::str::from_utf8(&line_bytes) {
            Ok(line) => line,
            Err(e) => {
                return vec![Err(Error::Stream(format!(
                    "Invalid UTF-8 in SSE stream: {e}"
                )))];
            }
        };

        self.process_line(line)
    }

    fn finalize_complete(&mut self) -> Vec<Result<StreamEvent>> {
        let mut events = Vec::new();

        if self.text_block_open {
            self.text_block_open = false;
            events.push(Ok(StreamEvent::BlockStop { index: 0 }));
        }

        if self.started {
            // Workers AI does not expose a reliable stream finish reason here,
            // so completion metadata stays intentionally partial.
            events.push(Ok(StreamEvent::ResponseMetadata {
                finish_reason: None,
                usage: None,
                usage_mode: UsageMetadataMode::Snapshot,
                id: None,
                model: None,
                metadata: ExtraMap::new(),
            }));
            events.push(Ok(StreamEvent::ResponseStop));
            self.started = false;
        }

        events
    }

    fn finalize_incomplete(&mut self) -> Vec<Result<StreamEvent>> {
        let mut events = Vec::new();

        if self.text_block_open {
            self.text_block_open = false;
            events.push(Ok(StreamEvent::BlockStop { index: 0 }));
        }

        self.started = false;
        events
    }

    fn finalize_on_eof(&mut self) -> (Vec<Result<StreamEvent>>, bool) {
        if self.finished {
            return (Vec::new(), false);
        }

        let mut events = self.drain_trailing_buffered_line();

        if self.finished {
            return (events, false);
        }

        let saw_response = self.started;
        events.extend(self.finalize_incomplete());
        (events, saw_response)
    }
}

/// A wrapper that makes a `!Send` `ByteStream` satisfy the `Send` bound.
///
/// Uses `SendWrapper` internally. This is safe in the Cloudflare Workers
/// environment because it is single-threaded — the `Send` bound on
/// `ChatStream` is never exercised at runtime.
struct SendChatStream {
    inner: SendWrapper<worker::ByteStream>,
    state: SseState,
    pending_events: VecDeque<Result<StreamEvent>>,
}

fn poll_chunk_stream<S, E>(
    mut inner: Pin<&mut S>,
    state: &mut SseState,
    pending_events: &mut VecDeque<Result<StreamEvent>>,
    cx: &mut Context<'_>,
) -> Poll<Option<Result<StreamEvent>>>
where
    S: Stream<Item = std::result::Result<Vec<u8>, E>>,
    E: fmt::Display,
{
    if let Some(event) = pending_events.pop_front() {
        return Poll::Ready(Some(event));
    }

    match inner.as_mut().poll_next(cx) {
        Poll::Ready(Some(Ok(bytes))) => {
            let events = state.process_bytes(&bytes);
            if events.is_empty() {
                cx.waker().wake_by_ref();
                Poll::Pending
            } else {
                *pending_events = events.into();
                Poll::Ready(pending_events.pop_front())
            }
        }
        Poll::Ready(Some(Err(err))) => {
            let (events, _) = state.finalize_on_eof();
            let mut events: VecDeque<_> = events.into();
            events.push_back(Err(Error::Stream(format!("ByteStream error: {err}"))));
            *pending_events = events;
            Poll::Ready(pending_events.pop_front())
        }
        Poll::Ready(None) => {
            let (events, saw_response) = state.finalize_on_eof();
            let mut events: VecDeque<_> = events.into();
            if saw_response {
                events.push_back(Err(Error::Stream("SSE stream ended before [DONE]".into())));
            }
            if events.is_empty() {
                Poll::Ready(None)
            } else {
                *pending_events = events;
                Poll::Ready(pending_events.pop_front())
            }
        }
        Poll::Pending => Poll::Pending,
    }
}

impl Stream for SendChatStream {
    type Item = Result<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        poll_chunk_stream(
            Pin::new(&mut *this.inner),
            &mut this.state,
            &mut this.pending_events,
            cx,
        )
    }
}

// SAFETY: SendWrapper<ByteStream> is Send. The Stream impl only accesses
// inner through &mut SendWrapper which is safe on the owning thread.
// In the Cloudflare Workers runtime, there is only one thread.
unsafe impl Send for SendChatStream {}

/// Convert a `worker::ByteStream` into an anyllm `ChatStream`.
pub(crate) fn byte_stream_to_chat_stream(byte_stream: worker::ByteStream) -> ChatStream {
    Box::pin(SendChatStream {
        inner: SendWrapper::new(byte_stream),
        state: SseState::new(),
        pending_events: VecDeque::new(),
    })
}

#[cfg(test)]
struct FixtureChatStream<S> {
    inner: S,
    state: SseState,
    pending_events: VecDeque<Result<StreamEvent>>,
}

#[cfg(test)]
impl<S, E> Stream for FixtureChatStream<S>
where
    S: Stream<Item = std::result::Result<Vec<u8>, E>> + Unpin,
    E: fmt::Display,
{
    type Item = Result<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        poll_chunk_stream(
            Pin::new(&mut this.inner),
            &mut this.state,
            &mut this.pending_events,
            cx,
        )
    }
}

#[cfg(test)]
pub(crate) fn conformance_stream_from_sse_text(text: &str) -> ChatStream {
    Box::pin(FixtureChatStream {
        inner: anyllm_conformance::byte_stream_from_sse_text(text),
        state: SseState::new(),
        pending_events: VecDeque::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyllm::StreamEvent;

    fn parse_raw_bytes(state: &mut SseState, bytes: &[u8]) -> Vec<Result<StreamEvent>> {
        state.process_bytes(bytes)
    }

    fn parse_bytes(state: &mut SseState, text: &str) -> Vec<Result<StreamEvent>> {
        parse_raw_bytes(state, text.as_bytes())
    }

    #[test]
    fn done_sentinel_emits_terminal_events_once() {
        let mut state = SseState::new();

        let text_events = parse_bytes(&mut state, "data: {\"response\":\"Hello\"}\n\n");
        assert!(
            text_events
                .iter()
                .filter_map(|event| event.as_ref().ok())
                .any(|event| matches!(event, StreamEvent::ResponseStart { .. }))
        );

        let done_events = parse_bytes(&mut state, "data: [DONE]\n\n");
        assert_eq!(
            done_events
                .iter()
                .filter_map(|event| event.as_ref().ok())
                .filter(|event| matches!(event, StreamEvent::ResponseMetadata { .. }))
                .count(),
            1
        );
        assert_eq!(
            done_events
                .iter()
                .filter_map(|event| event.as_ref().ok())
                .filter(|event| matches!(event, StreamEvent::ResponseStop))
                .count(),
            1
        );

        let (eof_events, saw_response) = state.finalize_on_eof();
        assert!(!saw_response);
        assert!(eof_events.is_empty());
    }

    #[test]
    fn eof_without_done_leaves_response_incomplete() {
        let mut state = SseState::new();

        assert!(
            parse_bytes(&mut state, "data: {\"response\":\"Hello\"}\n\n")
                .iter()
                .all(|event| event.is_ok())
        );

        let (eof_events, saw_response) = state.finalize_on_eof();
        assert!(saw_response);
        assert_eq!(eof_events.len(), 1);
        assert!(matches!(
            eof_events[0].as_ref().unwrap(),
            StreamEvent::BlockStop { index: 0 }
        ));
    }

    #[test]
    fn utf8_split_across_chunks_is_buffered_until_complete() {
        let mut state = SseState::new();
        let expected = String::from_utf8(vec![b'c', b'a', b'f', 0xC3, 0xA9]).unwrap();

        let prefix = b"data: {\"response\":\"caf";
        let first = [prefix.as_slice(), &[0xC3]].concat();
        assert!(parse_raw_bytes(&mut state, &first).is_empty());

        let second = [b"\xA9\"}\n\n".as_slice()].concat();
        let events = parse_raw_bytes(&mut state, &second);

        assert!(events.iter().all(|event| event.is_ok()));
        assert!(events.iter().filter_map(|event| event.as_ref().ok()).any(
            |event| matches!(event, StreamEvent::TextDelta { text, .. } if text == &expected)
        ));
    }
}
