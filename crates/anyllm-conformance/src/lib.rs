mod http_test_server;

use std::path::{Path, PathBuf};

use anyllm::{ChatResponse, ChatStream, StreamCollector, StreamEvent};
use futures::StreamExt;
use serde::Serialize;
use serde_json::{Value, json};

pub use http_test_server::{MockHttpResponse, RecordedRequest, TestHttpServer};

#[derive(Clone, Debug)]
pub struct FixtureDir {
    root: PathBuf,
}

impl FixtureDir {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn as_path(&self) -> &Path {
        &self.root
    }

    fn file(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
}

pub fn load_json_fixture(fixtures: &FixtureDir, name: &str) -> Value {
    let path = fixtures.file(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse fixture {} as json: {e}", path.display()))
}

pub fn load_text_fixture(fixtures: &FixtureDir, name: &str) -> String {
    let path = fixtures.file(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()))
}

pub fn assert_json_fixture_eq<T: Serialize>(actual: &T, fixtures: &FixtureDir, name: &str) {
    let actual = serde_json::to_value(actual).expect("serialize actual json fixture value");
    let expected = load_json_fixture(fixtures, name);
    assert_eq!(actual, expected, "json fixture mismatch: {name}");
}

pub fn assert_response_fixture_eq(actual: &ChatResponse, fixtures: &FixtureDir, name: &str) {
    let expected = load_json_fixture(fixtures, name);
    assert_eq!(
        actual.to_log_value(),
        expected,
        "response fixture mismatch: {name}"
    );
}

pub fn assert_error_fixture_eq(actual: &anyllm::Error, fixtures: &FixtureDir, name: &str) {
    let expected = load_json_fixture(fixtures, name);
    assert_eq!(
        serde_json::to_value(actual.as_log()).expect("serialize error log view"),
        expected,
        "error fixture mismatch: {name}"
    );
}

pub fn assert_event_results_fixture_eq(
    actual: &[anyllm::Result<StreamEvent>],
    fixtures: &FixtureDir,
    name: &str,
) {
    let actual: Vec<Value> = actual
        .iter()
        .map(|item| match item {
            Ok(event) => serde_json::to_value(event).expect("serialize stream event"),
            Err(err) => json!({"error": serde_json::to_value(err.as_log()).expect("serialize error log view")}),
        })
        .collect();
    let expected = load_json_fixture(fixtures, name);
    assert_eq!(
        Value::Array(actual),
        expected,
        "stream event fixture mismatch: {name}"
    );
}

pub fn byte_stream_from_sse_text(
    text: &str,
) -> impl futures::Stream<Item = Result<Vec<u8>, std::io::Error>> + Send + Unpin + 'static {
    let normalized = if text.ends_with("\n\n") {
        text.to_string()
    } else {
        format!("{text}\n\n")
    };
    futures::stream::iter(vec![Ok(normalized.into_bytes())])
}

async fn collect_stream_events(
    mut stream: ChatStream,
) -> (StreamCollector, Vec<anyllm::Result<StreamEvent>>) {
    let mut collector = StreamCollector::new();
    let mut seen = Vec::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => {
                collector
                    .push_ref(&event)
                    .expect("stream collector should accept fixture event");
                seen.push(Ok(event));
            }
            Err(err) => seen.push(Err(err)),
        }
    }

    (collector, seen)
}

pub async fn assert_stream_fixture_eq(
    stream: ChatStream,
    fixtures: &FixtureDir,
    events_name: &str,
    response_name: &str,
) {
    let (collector, seen) = collect_stream_events(stream).await;

    assert_event_results_fixture_eq(&seen, fixtures, events_name);

    let response = collector
        .finish()
        .expect("stream fixture should collect response");
    assert_response_fixture_eq(&response, fixtures, response_name);
}

pub async fn assert_partial_stream_fixture_eq(
    stream: ChatStream,
    fixtures: &FixtureDir,
    events_name: &str,
    response_name: &str,
    completeness_name: &str,
) {
    let (collector, seen) = collect_stream_events(stream).await;

    assert_event_results_fixture_eq(&seen, fixtures, events_name);

    let collected = collector
        .finish_partial()
        .expect("partial stream fixture should reconstruct response");
    assert_response_fixture_eq(&collected.response, fixtures, response_name);

    let expected = load_json_fixture(fixtures, completeness_name);
    let actual = serde_json::to_value(collected.completeness)
        .expect("serialize partial stream completeness");
    assert_eq!(
        actual, expected,
        "stream completeness fixture mismatch: {completeness_name}"
    );
}

pub async fn assert_stream_finish_error_fixture_eq(
    stream: ChatStream,
    fixtures: &FixtureDir,
    events_name: &str,
    error_name: &str,
) {
    let (collector, seen) = collect_stream_events(stream).await;

    assert_event_results_fixture_eq(&seen, fixtures, events_name);

    if let Some(err) = seen.iter().find_map(|item| item.as_ref().err()) {
        assert_error_fixture_eq(err, fixtures, error_name);
        return;
    }

    let err = collector
        .finish()
        .expect_err("stream fixture should fail strict collection");
    assert_error_fixture_eq(&err, fixtures, error_name);
}
