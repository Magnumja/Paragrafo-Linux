//! Answers questions about the open document with a local Llama model, grounded in passages retrieved from it.
//! Everything here blocks; the interface runs it on worker threads and applies the reported updates.

use crate::extraction::Paragraph;
use crate::library::{Library, SavedSource};
use crate::ollama::{self, ChatOptions, OllamaClient, OllamaError, OllamaMessage};
use crate::rag::{self, Chunk, Retrieved, VectorIndex};
use crate::text::{fold, prefix, sha256_hex};
use regex::Regex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

pub const EMBEDDING_MODEL: &str = "bge-m3";
pub const DEFAULT_CHAT_MODEL: &str = "llama3.2:3b";
pub const OVERVIEW_QUESTION: &str = "Faça uma síntese geral deste documento.";
/// Documents up to this many characters are synthesized in one pass; longer ones part by part.
pub const SINGLE_PASS_LIMIT: usize = 14000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Searching,
    Reading(usize, usize),
    Writing,
    Done,
    Failed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatMessage {
    pub id: u64,
    pub role: Role,
    pub text: String,
    pub sources: Vec<Retrieved>,
    pub cited: Vec<usize>,
    pub stage: Stage,
    pub is_overview: bool,
    pub note: Option<String>,
}

impl ChatMessage {
    pub fn new(id: u64, role: Role, text: &str, stage: Stage, is_overview: bool) -> Self {
        Self { id, role, text: text.into(), sources: Vec::new(), cited: Vec::new(), stage, is_overview, note: None }
    }

    pub fn apply(&mut self, update: Update) {
        match update {
            Update::Stage(stage) => self.stage = stage,
            Update::Sources(sources) => self.sources = sources,
            Update::Text(text) => self.text = text,
            Update::Finished { text, cited, note } => {
                self.text = text;
                self.cited = cited;
                self.note = note;
                self.stage = Stage::Done;
            }
        }
    }
}

/// What a running answer reports back.
#[derive(Clone, Debug, PartialEq)]
pub enum Update {
    Stage(Stage),
    Sources(Vec<Retrieved>),
    Text(String),
    Finished { text: String, cited: Vec<usize>, note: Option<String> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Availability {
    Unknown,
    Checking,
    Ready,
    ServerOff,
    Missing(Vec<String>),
    Pulling(String, f64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum IndexState {
    Empty,
    Indexing(usize, usize),
    Ready,
    Failed(String),
}

fn pattern(cell: &'static OnceLock<Regex>, source: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(source).expect("regex válida"))
}

/// A request for the whole document ("resuma o documento"), not for one part of it ("resuma a cláusula 5").
pub fn wants_overview(question: &str) -> bool {
    static PART: OnceLock<Regex> = OnceLock::new();
    static DOCUMENT: OnceLock<Regex> = OnceLock::new();
    static GENERAL: OnceLock<Regex> = OnceLock::new();
    static SUMMARY: OnceLock<Regex> = OnceLock::new();
    let folded = fold(question);
    if folded.chars().count() > 80
        || folded.chars().any(char::is_numeric)
        || pattern(&PART, "capitulo|secao|pagina|clausula|trecho|paragrafo|parte |artigo |item |tabela|anexo").is_match(&folded)
    {
        return false;
    }
    let words = folded.split(' ').filter(|word| !word.is_empty()).count();
    // "Resuma a política de garantia" names a topic; only a bare request or one about the document itself counts.
    let about_document = pattern(&DOCUMENT, "documento|texto|pdf|arquivo|relatorio|livro|artigo|contrato inteiro|tudo|document|file").is_match(&folded);
    if pattern(&GENERAL, "visao geral|overview|do que (se )?trata|sobre o que (e|fala)|pontos principais|principais pontos|main points").is_match(&folded) {
        return about_document || words <= 6;
    }
    if pattern(&SUMMARY, "resum|sintes|sintetiz|summar").is_match(&folded) {
        return about_document || words <= 3;
    }
    false
}

/// "E na Europa?" depends on the question before it; "Quem assina o contrato?" does not, however short.
pub fn is_follow_up(question: &str) -> bool {
    static OPENING: OnceLock<Regex> = OnceLock::new();
    let folded = fold(question);
    let folded = folded.trim();
    folded.split(' ').filter(|word| !word.is_empty()).count() <= 3
        || pattern(&OPENING, "^(e |mas |entao |e quanto|quanto a|nesse|nessa|nisso|disso|dele|dela|isso|ele |ela |esse |essa |and |what about)").is_match(folded)
}

/// Earlier answers go back to the model without their citation numbers (the passages are renumbered every
/// turn) and without whole-document syntheses, which would crowd out the passages.
pub fn history(messages: &[ChatMessage]) -> Vec<OllamaMessage> {
    static CITATION: OnceLock<Regex> = OnceLock::new();
    let citation = pattern(&CITATION, r"\s*\[\d+(?:\s*[,;\-]\s*\d+)*\]");
    let usable: Vec<&ChatMessage> = messages.iter().filter(|m| m.stage == Stage::Done && !m.text.is_empty() && !m.is_overview).collect();
    usable[usable.len().saturating_sub(4)..]
        .iter()
        .map(|message| {
            let role = if message.role == Role::User { "user" } else { "assistant" };
            OllamaMessage::new(role, prefix(&citation.replace_all(&message.text, ""), 600))
        })
        .collect()
}

/// The last real question, so a short follow-up can be searched next to it.
pub fn previous_question(messages: &[ChatMessage]) -> Option<String> {
    messages.iter().rev().find(|m| m.role == Role::User && !wants_overview(&m.text)).map(|m| m.text.clone())
}

#[derive(Clone, Debug)]
pub struct CheckResult {
    pub availability: Availability,
    /// Every local model, including the ones that only embed or translate.
    pub installed: Vec<String>,
    pub chat_models: Vec<String>,
    pub chat_model: String,
}

/// Is Ollama there, and are the models? Starts the server when asked to and it is installed.
pub fn check(client: &OllamaClient, chat_model: &str, launch: bool) -> CheckResult {
    let off = |chat_model: &str| CheckResult { availability: Availability::ServerOff, installed: Vec::new(), chat_models: Vec::new(), chat_model: chat_model.to_string() };
    let mut running = client.is_running();
    if !running && launch && ollama::launch_server() {
        for _ in 0..25 {
            std::thread::sleep(Duration::from_millis(400));
            running = client.is_running();
            if running {
                break;
            }
        }
    }
    if !running {
        return off(chat_model);
    }
    // A failed listing says nothing about which models exist; do not touch the saved choice.
    let Ok(installed) = client.models() else { return off(chat_model) };
    let mut chat_models: Vec<String> = installed
        .iter()
        .filter(|name| !["embed", "bge", "minilm", "e5", "translate"].iter().any(|kind| name.to_lowercase().contains(kind)))
        .cloned()
        .collect();
    chat_models.sort();
    let chat_model = if ollama::has_model(chat_model, &chat_models) {
        chat_model.to_string()
    } else {
        chat_models.iter().find(|name| name.to_lowercase().contains("llama")).or(chat_models.first()).cloned().unwrap_or_else(|| DEFAULT_CHAT_MODEL.to_string())
    };
    let mut missing = Vec::new();
    if !ollama::has_model(EMBEDDING_MODEL, &installed) {
        missing.push(EMBEDDING_MODEL.to_string());
    }
    if chat_models.is_empty() {
        missing.push(DEFAULT_CHAT_MODEL.to_string());
    }
    let availability = if missing.is_empty() { Availability::Ready } else { Availability::Missing(missing) };
    CheckResult { availability, installed, chat_models, chat_model }
}

/// Vectors are only reused for exactly the text they were computed from.
pub fn index_key(chunks: &[Chunk]) -> String {
    let digest = sha256_hex(chunks.iter().map(|chunk| chunk.text.as_str()).collect::<Vec<_>>().join("\u{1F}").as_bytes());
    format!("{EMBEDDING_MODEL}|chunker-1|{}|{digest}", chunks.len())
}

/// Restores the saved search index or embeds every chunk and saves the result.
pub fn build_index(
    client: &OllamaClient,
    library: &Library,
    fingerprint: &str,
    chunks: &[Chunk],
    cancel: &AtomicBool,
    mut progress: impl FnMut(usize, usize),
) -> Result<VectorIndex, OllamaError> {
    let key = index_key(chunks);
    if let Some(saved) = library.load(fingerprint).filter(|saved| saved.index_key.as_deref() == Some(&key)) {
        if let Some(vectors) = saved.dimension.and_then(|dimension| library.load_vectors(fingerprint, chunks.len(), dimension)) {
            return Ok(VectorIndex::new(chunks.to_vec(), &vectors));
        }
    }
    progress(0, chunks.len());
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
    for batch in chunks.chunks(16) {
        if cancel.load(Ordering::Relaxed) {
            return Err(OllamaError::Cancelled);
        }
        let texts: Vec<String> = batch.iter().map(|chunk| chunk.text.clone()).collect();
        let embedded = client.embed(EMBEDDING_MODEL, &texts, cancel)?;
        if embedded.len() != texts.len() {
            return Err(OllamaError::Server("O modelo de busca devolveu uma resposta incompleta.".into()));
        }
        vectors.extend(embedded);
        progress(vectors.len(), chunks.len());
    }
    if cancel.load(Ordering::Relaxed) {
        return Err(OllamaError::Cancelled);
    }
    let index = VectorIndex::new(chunks.to_vec(), &vectors);
    library.save_vectors(&vectors, fingerprint);
    library.update(fingerprint, |saved| {
        saved.index_key = Some(key);
        saved.dimension = vectors.first().map(Vec::len);
    });
    Ok(index)
}

/// Streams a reply, reporting the text so far at most ~15 times a second.
fn stream(client: &OllamaClient, model: &str, messages: &[OllamaMessage], cancel: &AtomicBool, emit: &mut dyn FnMut(Update)) -> Result<String, OllamaError> {
    let mut text = String::new();
    let mut shown = Instant::now();
    // Temperature 0: the answer should follow the passages, not vary between runs.
    let options = ChatOptions { temperature: 0.0, ..ChatOptions::default() };
    client.chat(model, messages, options, cancel, |delta| {
        text.push_str(delta);
        if shown.elapsed() > Duration::from_millis(66) {
            shown = Instant::now();
            emit(Update::Text(text.clone()));
        }
    })?;
    if cancel.load(Ordering::Relaxed) {
        return Err(OllamaError::Cancelled);
    }
    Ok(text.trim().to_string())
}

pub struct Question<'a> {
    pub text: &'a str,
    pub previous: Option<&'a str>,
    pub history: &'a [OllamaMessage],
    /// Translations by paragraph index; they join the word search so a question in the target language finds the passage.
    pub translations: &'a [String],
}

/// Retrieves the passages for a question and writes an answer that cites them.
pub fn answer(
    client: &OllamaClient,
    index: &VectorIndex,
    chat_model: &str,
    question: &Question,
    cancel: &AtomicBool,
    emit: &mut dyn FnMut(Update),
) -> Result<(), OllamaError> {
    // A short follow-up ("e na Europa?") only makes sense next to the question before it.
    let query = match question.previous {
        Some(previous) if is_follow_up(question.text) => format!("{previous} {}", question.text),
        _ => question.text.to_string(),
    };
    let vector = client
        .embed(EMBEDDING_MODEL, std::slice::from_ref(&query), cancel)?
        .into_iter()
        .next()
        .ok_or_else(|| OllamaError::Server("Não foi possível interpretar a pergunta.".into()))?;
    let mut extra: HashMap<usize, String> = HashMap::new();
    if !question.translations.is_empty() {
        for chunk in &index.chunks {
            let text = chunk.paragraphs.clone().filter_map(|at| question.translations.get(at)).filter(|t| !t.is_empty()).cloned().collect::<Vec<_>>().join(" ");
            if !text.is_empty() {
                extra.insert(chunk.id, text);
            }
        }
    }
    let sources = index.search(&vector, &query, &extra, rag::SEARCH_LIMIT, rag::MINIMUM_SCORE);
    if cancel.load(Ordering::Relaxed) {
        return Err(OllamaError::Cancelled);
    }
    if sources.is_empty() {
        emit(Update::Finished {
            text: "Não encontrei nenhum trecho do documento relacionado a essa pergunta. Tente reformular ou usar termos que aparecem no texto.".into(),
            cited: Vec::new(),
            note: None,
        });
        return Ok(());
    }
    emit(Update::Sources(sources.clone()));
    emit(Update::Stage(Stage::Writing));
    let text = stream(client, chat_model, &rag::messages(question.text, &sources, question.history), cancel, emit)?;
    let cited = rag::citations(&text, sources.len());
    let note = cited.is_empty().then(|| "A resposta não citou um trecho específico. Confira nos trechos consultados.".to_string());
    emit(Update::Finished { text, cited, note });
    Ok(())
}

pub struct Overview<'a> {
    pub fingerprint: &'a str,
    pub paragraphs: &'a [Paragraph],
    pub chunks: &'a [Chunk],
    pub chat_model: &'a str,
    pub question: &'a str,
    /// Discard the saved synthesis and write a new one.
    pub force: bool,
    pub single_pass_limit: usize,
}

/// Whole-document synthesis. Short documents go to the model at once; long ones are read in parts and then combined.
/// Either way the model writes from numbered passages, so every point links back to where it came from.
pub fn overview(client: &OllamaClient, library: &Library, request: &Overview, cancel: &AtomicBool, emit: &mut dyn FnMut(Update)) -> Result<(), OllamaError> {
    let standard = request.question == OVERVIEW_QUESTION;
    if !request.force && standard {
        if let Some(saved) = library.load(request.fingerprint) {
            if let Some(text) = saved.summary.filter(|text| !text.is_empty()) {
                let sources: Vec<Retrieved> = saved
                    .summary_sources
                    .unwrap_or_default()
                    .into_iter()
                    .enumerate()
                    .map(|(position, source)| Retrieved {
                        chunk: Chunk { id: position, page: source.page, paragraphs: source.paragraph..source.paragraph + 1, text: source.text },
                        score: 0.0,
                    })
                    .collect();
                let cited = rag::citations(&text, sources.len());
                emit(Update::Sources(sources));
                emit(Update::Finished { text, cited, note: Some("Síntese salva deste documento.".into()) });
                return Ok(());
            }
        }
    }

    let mut note = None;
    let mut sources: Vec<Retrieved> = Vec::new();
    let total_length: usize = request.chunks.iter().map(|chunk| chunk.text.chars().count()).sum();
    if total_length <= request.single_pass_limit {
        // Smaller passages than the search index uses, so each point has its own passage to cite.
        let fine = rag::chunks_with(request.paragraphs, 320, 700);
        let chosen = if fine.len() <= 40 { fine } else { request.chunks.to_vec() };
        sources = chosen.into_iter().map(|chunk| Retrieved { chunk, score: 0.0 }).collect();
    } else {
        struct Part<'a> {
            first: &'a Chunk,
            last: usize,
            text: String,
            length: usize,
        }
        let mut parts: Vec<Part> = Vec::new();
        for chunk in request.chunks {
            let labelled = format!("[página {}]\n{}", chunk.page, chunk.text);
            let size = labelled.chars().count();
            match parts.last_mut() {
                Some(current) if current.length + size <= request.single_pass_limit / 2 => {
                    current.last = chunk.page;
                    current.text = format!("{}\n\n{labelled}", current.text);
                    current.length += size + 2;
                }
                _ => parts.push(Part { first: chunk, last: chunk.page, text: labelled, length: size }),
            }
        }
        let limit = 24;
        if parts.len() > limit {
            let total = parts.len();
            let mut kept: Vec<Option<Part>> = parts.into_iter().map(Some).collect();
            parts = (0..limit).filter_map(|step| kept[step * (total - 1) / (limit - 1)].take()).collect();
            note = Some(format!("Documento longo: síntese feita a partir de {limit} de {total} partes, distribuídas do início ao fim."));
        }
        let map = format!(
            "Você resume um trecho de um documento. Escreva no mesmo idioma desta solicitação: «{}». Escreva de 3 a 5 frases objetivas com os fatos, números e conclusões do trecho. Não invente nada.",
            request.question
        );
        let count = parts.len();
        for (step, part) in parts.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err(OllamaError::Cancelled);
            }
            emit(Update::Stage(Stage::Reading(step + 1, count)));
            let mut text = String::new();
            let options = ChatOptions { temperature: 0.0, ..ChatOptions::default() };
            let messages = [OllamaMessage::new("system", map.as_str()), OllamaMessage::new("user", part.text.as_str())];
            client.chat(request.chat_model, &messages, options, cancel, |delta| text.push_str(delta))?;
            if cancel.load(Ordering::Relaxed) {
                return Err(OllamaError::Cancelled);
            }
            let pages = if part.first.page == part.last {
                format!("Resumo automático da página {}: ", part.last)
            } else {
                format!("Resumo automático das páginas {}–{}: ", part.first.page, part.last)
            };
            sources.push(Retrieved {
                chunk: Chunk { id: step, page: part.first.page, paragraphs: part.first.paragraphs.clone(), text: pages + text.trim() },
                score: 0.0,
            });
        }
    }
    emit(Update::Sources(sources.clone()));
    emit(Update::Stage(Stage::Writing));
    let ask = format!(
        "{} Escreva primeiro um parágrafo de 2 a 3 frases dizendo do que o documento trata. Depois liste de 4 a 7 pontos principais, um por linha começando com \"- \", cada um com a citação do trecho de onde veio.",
        request.question
    );
    let text = stream(client, request.chat_model, &rag::messages(&ask, &sources, &[]), cancel, emit)?;
    let cited = rag::citations(&text, sources.len());
    emit(Update::Finished { text: text.clone(), cited, note });
    if !text.is_empty() && standard {
        library.update(request.fingerprint, |saved| {
            saved.summary = Some(text);
            saved.summary_sources = Some(
                sources
                    .iter()
                    .map(|source| SavedSource { page: source.chunk.page, paragraph: source.chunk.paragraphs.start, text: prefix(&source.chunk.text, 400).to_string() })
                    .collect(),
            );
        });
    }
    Ok(())
}
