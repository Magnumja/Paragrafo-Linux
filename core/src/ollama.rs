//! HTTP client for the local Ollama server. Loopback only: nothing leaves the computer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq)]
pub enum OllamaError {
    LocalOnly,
    NotRunning,
    ModelMissing(String),
    Server(String),
    Cancelled,
}

impl std::fmt::Display for OllamaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalOnly => f.write_str("Somente modelos e servidor locais são permitidos."),
            Self::NotRunning => f.write_str("O Ollama não está em execução."),
            Self::ModelMissing(model) if model.is_empty() => f.write_str("Modelo não instalado no Ollama."),
            Self::ModelMissing(model) => write!(f, "O modelo “{model}” não está instalado no Ollama."),
            Self::Server(message) if message.is_empty() => f.write_str("O Ollama retornou um erro."),
            Self::Server(message) => write!(f, "Erro do Ollama: {message}"),
            Self::Cancelled => f.write_str("Operação interrompida."),
        }
    }
}

impl std::error::Error for OllamaError {}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

impl OllamaMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into() }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ChatOptions {
    pub temperature: f64,
    pub context_length: usize,
    pub max_tokens: usize,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self { temperature: 0.2, context_length: 8192, max_tokens: 1024 }
    }
}

#[derive(Clone, Debug)]
pub struct OllamaClient {
    base: String,
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self::new("http://127.0.0.1:11434")
    }
}

const KEEP_ALIVE: &str = "2m";
const EMBED_BATCH: usize = 32;

/// Short requests: each call sets its own deadline.
fn agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    // A local server must never redirect a document request to another host, and no proxy is ever consulted.
    AGENT.get_or_init(|| ureq::AgentBuilder::new().redirects(0).try_proxy_from_env(false).timeout_connect(Duration::from_secs(3)).build())
}

/// Streams: no overall deadline (a model download takes as long as it takes), only an idle one.
fn stream_agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::AgentBuilder::new()
            .redirects(0)
            .try_proxy_from_env(false)
            .timeout_connect(Duration::from_secs(3))
            .timeout_read(Duration::from_secs(600))
            .build()
    })
}

impl OllamaClient {
    pub fn new(base: &str) -> Self {
        Self { base: base.trim_end_matches('/').to_string() }
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{path}", self.base)
    }

    // Public API

    pub fn is_running(&self) -> bool {
        if !is_local_endpoint(&self.base) {
            return false;
        }
        matches!(agent().get(&self.url("api/version")).timeout(Duration::from_secs(2)).call(), Ok(response) if response.status() == 200)
    }

    /// Installed local models. Cloud-backed entries are left out.
    pub fn models(&self) -> Result<Vec<String>, OllamaError> {
        if !is_local_endpoint(&self.base) {
            return Err(OllamaError::LocalOnly);
        }
        let response = agent().get(&self.url("api/tags")).timeout(Duration::from_secs(10)).call().map_err(|error| mapped(error, None))?;
        let status = response.status();
        if status != 200 {
            return Err(OllamaError::Server(format!("HTTP {status}")));
        }
        let tags: TagsResponse = response.into_json().map_err(|_| OllamaError::Server("Resposta inesperada ao listar modelos.".into()))?;
        Ok(tags
            .models
            .into_iter()
            .filter(|model| model.remote_host.is_none() && model.remote_model.is_none() && is_local_model(&model.name))
            .map(|model| model.name)
            .collect())
    }

    /// Unit-length embeddings, one per input, in order.
    pub fn embed(&self, model: &str, inputs: &[String], cancel: &AtomicBool) -> Result<Vec<Vec<f32>>, OllamaError> {
        if !is_local_model(model) || !is_local_endpoint(&self.base) {
            return Err(OllamaError::LocalOnly);
        }
        let mut result = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(EMBED_BATCH) {
            if cancel.load(Ordering::Relaxed) {
                return Err(OllamaError::Cancelled);
            }
            let body = EmbedRequest { model, input: batch, truncate: true, keep_alive: KEEP_ALIVE };
            let response = agent()
                .post(&self.url("api/embed"))
                .timeout(Duration::from_secs(300))
                .send_json(&body)
                .map_err(|error| mapped(error, Some(model)))?;
            let status = response.status();
            if status != 200 {
                return Err(OllamaError::Server(format!("HTTP {status}")));
            }
            let unexpected = || OllamaError::Server("Resposta inesperada ao gerar embeddings.".into());
            let decoded: EmbedResponse = response.into_json().map_err(|_| unexpected())?;
            let dimension = decoded.embeddings.first().map_or(0, Vec::len);
            if decoded.embeddings.len() != batch.len() || dimension == 0 || decoded.embeddings.iter().any(|vector| vector.len() != dimension) {
                return Err(unexpected());
            }
            result.extend(decoded.embeddings.into_iter().map(normalized));
        }
        Ok(result)
    }

    /// Streams the reply; `on_delta` receives each piece as it arrives. Setting `cancel` ends the stream at the next piece.
    pub fn chat(
        &self,
        model: &str,
        messages: &[OllamaMessage],
        options: ChatOptions,
        cancel: &AtomicBool,
        mut on_delta: impl FnMut(&str),
    ) -> Result<(), OllamaError> {
        let body = ChatRequest {
            model,
            messages,
            stream: true,
            options: ChatRequestOptions {
                temperature: options.temperature,
                num_ctx: options.context_length.clamp(512, 8192),
                num_predict: options.max_tokens,
            },
            keep_alive: KEEP_ALIVE,
        };
        self.stream("api/chat", &body, model, cancel, |line: ChatLine| {
            if let Some(delta) = line.message.and_then(|message| message.content).filter(|delta| !delta.is_empty()) {
                on_delta(&delta);
            }
            line.done == Some(true)
        })
    }

    /// Downloads a model; `on_progress` receives a monotonic 0...1 fraction.
    pub fn pull(&self, model: &str, cancel: &AtomicBool, mut on_progress: impl FnMut(f64)) -> Result<(), OllamaError> {
        let mut progress = PullProgress::default();
        self.stream("api/pull", &PullRequest { model, stream: true }, model, cancel, |line: PullLine| {
            if line.status.as_deref() == Some("success") {
                on_progress(1.0);
                return true;
            }
            if let Some(value) = progress.update(&line) {
                on_progress(value);
            }
            false
        })
    }

    // Internals

    /// One NDJSON object per line; `handle` returns true when the stream is complete.
    fn stream<Body: Serialize, Line: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &Body,
        model: &str,
        cancel: &AtomicBool,
        mut handle: impl FnMut(Line) -> bool,
    ) -> Result<(), OllamaError> {
        if !is_local_model(model) || !is_local_endpoint(&self.base) {
            return Err(OllamaError::LocalOnly);
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(OllamaError::Cancelled);
        }
        let response = stream_agent().post(&self.url(path)).send_json(body).map_err(|error| mapped(error, Some(model)))?;
        let status = response.status();
        if status != 200 {
            return Err(OllamaError::Server(format!("HTTP {status}")));
        }
        // Dropping the reader closes the connection, which is how the server learns to stop generating.
        for line in BufReader::new(response.into_reader()).lines() {
            if cancel.load(Ordering::Relaxed) {
                return Err(OllamaError::Cancelled);
            }
            let line = line.map_err(|error| transport(&error.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(failure) = serde_json::from_str::<ErrorLine>(&line) {
                return Err(failure_from(&failure.error, 200, Some(model)));
            }
            let Ok(decoded) = serde_json::from_str::<Line>(&line) else { continue };
            if handle(decoded) {
                break;
            }
        }
        Ok(())
    }
}

/// True when `wanted` is installed: without a tag it matches any tag, with a tag it must match exactly.
pub fn has_model(wanted: &str, installed: &[String]) -> bool {
    let wanted = wanted.trim().to_lowercase();
    if wanted.is_empty() {
        return false;
    }
    let tagged = has_tag(&wanted);
    installed.iter().any(|name| {
        let name = name.to_lowercase();
        if name == wanted {
            return true;
        }
        !tagged && has_tag(&name) && name.rfind(':').is_some_and(|colon| name[..colon] == wanted)
    })
}

/// A tag is a colon after the last path separator ("host:5000/model" has none).
fn has_tag(name: &str) -> bool {
    name.rsplit('/').next().unwrap_or(name).contains(':')
}

/// Plain HTTP to the loopback address, with nothing else in the URL.
pub fn is_local_endpoint(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("http://") else { return false };
    if rest.contains(['@', '?', '#']) {
        return false;
    }
    let (authority, path) = rest.split_once('/').map_or((rest, ""), |(authority, path)| (authority, path));
    if !path.is_empty() {
        return false;
    }
    let (host, port) = if let Some(end) = authority.find(']') {
        (&authority[..=end], authority[end + 1..].strip_prefix(':'))
    } else {
        authority.split_once(':').map_or((authority, None), |(host, port)| (host, Some(port)))
    };
    let port_ok = port.is_none_or(|port| !port.is_empty() && port.len() <= 5 && port.bytes().all(|b| b.is_ascii_digit()));
    (host == "127.0.0.1" || host == "[::1]") && port_ok && (port.is_some() || authority == host)
}

pub fn is_local_model(name: &str) -> bool {
    !name.is_empty() && !name.to_lowercase().contains("cloud") && !name.contains("://")
}

/// Starts `ollama serve` when the binary is installed. Returns whether something was started.
pub fn launch_server() -> bool {
    let mut candidates: Vec<std::path::PathBuf> = ["/usr/local/bin/ollama", "/usr/bin/ollama", "/opt/ollama/bin/ollama", "/snap/bin/ollama"]
        .iter()
        .map(std::path::PathBuf::from)
        .collect();
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(std::path::PathBuf::from(home).join(".local/bin/ollama"));
    }
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path).filter(|dir| dir.is_absolute()).map(|dir| dir.join("ollama")));
    }
    candidates.into_iter().filter(|path| path.is_file()).any(|path| {
        Command::new(path).arg("serve").stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().is_ok()
    })
}

fn normalized(vector: Vec<f32>) -> Vec<f32> {
    if !vector.iter().all(|value| value.is_finite()) {
        return vec![0.0; vector.len()];
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.into_iter().map(|value| value / norm).collect()
    } else {
        vector
    }
}

fn failure_from(message: &str, status: u16, model: Option<&str>) -> OllamaError {
    let lowered = message.to_lowercase();
    let missing = status == 404 || (lowered.contains("model") && lowered.contains("not found")) || lowered.contains("manifest: file does not exist");
    match model {
        Some(model) if missing => OllamaError::ModelMissing(model.to_string()),
        _ => OllamaError::Server(message.to_string()),
    }
}

fn transport(message: &str) -> OllamaError {
    let lowered = message.to_lowercase();
    if lowered.contains("timed out") || lowered.contains("timeout") {
        OllamaError::Server("Tempo de resposta esgotado.".into())
    } else {
        OllamaError::NotRunning
    }
}

fn mapped(error: ureq::Error, model: Option<&str>) -> OllamaError {
    match error {
        ureq::Error::Status(status, response) => {
            let mut body = String::new();
            let _ = response.into_reader().take(65_536).read_to_string(&mut body);
            let message = serde_json::from_str::<ErrorLine>(&body).map(|line| line.error).unwrap_or_else(|_| {
                let trimmed: String = body.trim().chars().take(300).collect();
                trimmed
            });
            failure_from(if message.is_empty() { "erro sem descrição" } else { &message }, status, model)
        }
        ureq::Error::Transport(transport_error) => match transport_error.kind() {
            ureq::ErrorKind::ConnectionFailed | ureq::ErrorKind::Dns => OllamaError::NotRunning,
            ureq::ErrorKind::Io => transport(&transport_error.to_string()),
            _ => OllamaError::Server(transport_error.to_string()),
        },
    }
}

// Wire formats

#[derive(Deserialize)]
struct TagsResponse {
    models: Vec<TagModel>,
}

#[derive(Deserialize)]
struct TagModel {
    name: String,
    #[serde(default)]
    remote_model: Option<String>,
    #[serde(default)]
    remote_host: Option<String>,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    truncate: bool,
    keep_alive: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct ChatRequestOptions {
    temperature: f64,
    num_ctx: usize,
    num_predict: usize,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [OllamaMessage],
    stream: bool,
    options: ChatRequestOptions,
    keep_alive: &'a str,
}

#[derive(Deserialize)]
struct ChatLineMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct ChatLine {
    #[serde(default)]
    message: Option<ChatLineMessage>,
    #[serde(default)]
    done: Option<bool>,
}

#[derive(Serialize)]
struct PullRequest<'a> {
    model: &'a str,
    stream: bool,
}

#[derive(Deserialize)]
struct PullLine {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    digest: Option<String>,
    #[serde(default)]
    total: Option<f64>,
    #[serde(default)]
    completed: Option<f64>,
}

#[derive(Deserialize)]
struct ErrorLine {
    error: String,
}

/// Aggregates per-layer byte counts into a monotonic 0...1 fraction.
#[derive(Default)]
struct PullProgress {
    layers: HashMap<String, (f64, f64)>,
    last: Option<f64>,
}

impl PullProgress {
    fn update(&mut self, line: &PullLine) -> Option<f64> {
        let (Some(digest), Some(total)) = (&line.digest, line.total.filter(|total| *total > 0.0)) else {
            return if self.last.is_none() { self.emit(0.0) } else { None };
        };
        self.layers.insert(digest.clone(), (total, line.completed.unwrap_or(0.0).clamp(0.0, total)));
        let all: f64 = self.layers.values().map(|layer| layer.0).sum();
        let done: f64 = self.layers.values().map(|layer| layer.1).sum();
        // Layers are announced one at a time, so cap below 1 until "success".
        self.emit((done / all).min(0.99))
    }

    fn emit(&mut self, value: f64) -> Option<f64> {
        if self.last.is_some_and(|last| value <= last) {
            return None;
        }
        self.last = Some(value);
        Some(value)
    }
}
