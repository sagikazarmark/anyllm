use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::{Request, State};
use axum::response::Response;
use axum::routing::any;
use serde::Serialize;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

#[derive(Debug, Clone)]
pub struct MockHttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl MockHttpResponse {
    #[must_use]
    pub fn text(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: body.into(),
        }
    }

    #[must_use]
    pub fn json(status: u16, value: &impl Serialize) -> Self {
        Self::text(
            status,
            serde_json::to_string(value).expect("serialize mock JSON response"),
        )
        .with_header("content-type", "application/json")
    }

    #[must_use]
    pub fn sse(body: impl Into<String>) -> Self {
        let body = body.into();
        let normalized = if body.ends_with("\n\n") {
            body
        } else {
            format!("{body}\n\n")
        };

        Self::text(200, normalized).with_header("content-type", "text/event-stream")
    }

    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

#[derive(Debug, Clone)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub query: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

impl RecordedRequest {
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    pub fn body_text(&self) -> &str {
        std::str::from_utf8(&self.body).expect("recorded request body should be UTF-8")
    }

    pub fn body_json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).expect("recorded request body should be valid JSON")
    }
}

#[derive(Clone)]
struct AppState {
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    responses: Arc<Mutex<VecDeque<MockHttpResponse>>>,
}

fn header_map_to_btree(headers: &axum::http::HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_ascii_lowercase(), value.to_string()))
        })
        .collect()
}

async fn handle_request(State(state): State<AppState>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, usize::MAX)
        .await
        .expect("read mock server request body");

    state.requests.lock().await.push(RecordedRequest {
        method: parts.method.to_string(),
        path: parts.uri.path().to_string(),
        query: parts.uri.query().map(str::to_string),
        headers: header_map_to_btree(&parts.headers),
        body: body.to_vec(),
    });

    let response = state
        .responses
        .lock()
        .await
        .pop_front()
        .expect("mock server ran out of queued responses");

    let mut builder = Response::builder().status(response.status);
    for (name, value) in response.headers {
        builder = builder.header(name, value);
    }

    builder
        .body(Body::from(response.body))
        .expect("build mock HTTP response")
}

pub struct TestHttpServer {
    base_url: String,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    task: JoinHandle<()>,
}

impl TestHttpServer {
    pub async fn spawn(responses: impl IntoIterator<Item = MockHttpResponse>) -> Self {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let responses = Arc::new(Mutex::new(VecDeque::from_iter(responses)));
        let state = AppState {
            requests: Arc::clone(&requests),
            responses,
        };

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock HTTP server listener");
        let base_url = format!(
            "http://{}",
            listener.local_addr().expect("local listener addr")
        );
        let app = Router::new()
            .fallback(any(handle_request))
            .with_state(state);

        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        Self {
            base_url,
            requests,
            task,
        }
    }

    #[must_use]
    pub fn url(&self) -> &str {
        &self.base_url
    }

    pub async fn recorded_requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().await.clone()
    }
}

impl Drop for TestHttpServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}
