//! Translation with a local model. Linux has no system translation engine, so the same Ollama
//! that answers questions translates the paragraphs; nothing leaves the computer.

use crate::language::english_name;
use crate::ollama::{has_model, ChatOptions, OllamaClient, OllamaError, OllamaMessage};
use regex::Regex;
use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;

/// A model made for translation; used when installed (`ollama pull translategemma:4b`).
pub const PREFERRED_MODEL: &str = "translategemma";

/// `$PARAGRAFO_TRANSLATION_MODEL` when it is installed, else a translation model, else the chat model.
pub fn pick_model(installed: &[String], chat_models: &[String], chat_model: &str) -> Option<String> {
    if let Ok(wanted) = std::env::var("PARAGRAFO_TRANSLATION_MODEL") {
        if has_model(&wanted, installed) {
            return Some(wanted);
        }
    }
    if let Some(dedicated) = installed.iter().find(|name| name.to_lowercase().contains(PREFERRED_MODEL)) {
        return Some(dedicated.clone());
    }
    has_model(chat_model, chat_models).then(|| chat_model.to_string()).or_else(|| chat_models.first().cloned())
}

/// The prompt TranslateGemma was trained on; general chat models follow it just as well.
pub fn prompt(source: &str, target: &str, text: &str) -> Vec<OllamaMessage> {
    let (from, to) = (english_name(source), english_name(target));
    vec![OllamaMessage::new(
        "user",
        format!(
            "You are a professional {from} ({source}) to {to} ({target}) translator. Your goal is to accurately convey the meaning and nuances of the original {from} text while adhering to {to} grammar, vocabulary, and cultural sensitivities.\nProduce only the {to} translation, without any additional explanations or commentary. Keep numbers, names and references exactly as they are. Please translate the following {from} text into {to}:\n\n\n{text}"
        ),
    )]
}

fn regex(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("regex válida"))
}

/// What a model adds around the translation that the original never had: reasoning blocks of
/// "thinking" models, a "Here is the translation:" line, a code fence, wrapping quotes.
pub fn clean(output: &str, original: &str) -> String {
    static THINKING: OnceLock<Regex> = OnceLock::new();
    static PREAMBLE: OnceLock<Regex> = OnceLock::new();
    let without_thinking = regex(&THINKING, r"(?s)<think>.*?(</think>|$)").replace_all(output, "");
    let mut text = without_thinking.trim();
    // Only a whole first line that introduces the answer, and only when an answer follows it.
    let preamble = regex(
        &PREAMBLE,
        r"(?i)^(here is|here's|below is|sure[,!]?|aqui está|aqui esta|segue|claro[,!]?|tradução|traducao|translation|traducción)[^\n]{0,80}:\s*\n",
    );
    if let Some(found) = preamble.find(text) {
        let rest = text[found.end()..].trim();
        if !rest.is_empty() {
            text = rest;
        }
    }
    if let Some(inner) = text.strip_prefix("```").and_then(|rest| rest.strip_suffix("```")) {
        text = inner.trim_start_matches(|c: char| c.is_alphanumeric()).trim();
    }
    for (open, close) in [('"', '"'), ('“', '”'), ('«', '»')] {
        let wrapped = text.chars().count() >= 2 && text.starts_with(open) && text.ends_with(close);
        if wrapped && !original.trim().starts_with(open) {
            text = text[open.len_utf8()..text.len() - close.len_utf8()].trim();
        }
    }
    text.to_string()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Translation {
    pub text: String,
    /// The model gave nothing usable twice; the paragraph keeps its original text so the rest can go on.
    pub kept_original: bool,
}

fn attempt(client: &OllamaClient, model: &str, source: &str, target: &str, text: &str, temperature: f64, cancel: &AtomicBool) -> Result<String, OllamaError> {
    let mut output = String::new();
    // Room for the answer grows with the paragraph.
    let options = ChatOptions { temperature, context_length: 8192, max_tokens: (text.chars().count() * 2).clamp(256, 4096) };
    client.chat(model, &prompt(source, target, text), options, cancel, |delta| output.push_str(delta))?;
    Ok(clean(&output, text))
}

/// Translates one paragraph. An empty answer is retried once with a little variation;
/// a second empty answer keeps the original instead of stopping the whole document.
pub fn translate(client: &OllamaClient, model: &str, source: &str, target: &str, text: &str, cancel: &AtomicBool) -> Result<Translation, OllamaError> {
    for temperature in [0.0, 0.3] {
        let translated = attempt(client, model, source, target, text, temperature, cancel)?;
        if !translated.is_empty() {
            return Ok(Translation { text: translated, kept_original: false });
        }
    }
    Ok(Translation { text: text.to_string(), kept_original: true })
}
