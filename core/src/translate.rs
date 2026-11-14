//! Translation with a local model. Linux has no system translation engine, so the same Ollama
//! that answers questions translates the paragraphs; nothing leaves the computer.

use crate::language::english_name;
use crate::ollama::{has_model, ChatOptions, OllamaClient, OllamaError, OllamaMessage};
use std::sync::atomic::AtomicBool;

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

/// Models sometimes wrap the translation in quotes or a code fence the original never had.
pub fn clean(output: &str, original: &str) -> String {
    let mut text = output.trim();
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

/// Translates one paragraph. Room for the answer grows with the paragraph.
pub fn translate(client: &OllamaClient, model: &str, source: &str, target: &str, text: &str, cancel: &AtomicBool) -> Result<String, OllamaError> {
    let mut output = String::new();
    let options = ChatOptions { temperature: 0.0, context_length: 8192, max_tokens: (text.chars().count() * 2).clamp(256, 4096) };
    client.chat(model, &prompt(source, target, text), options, cancel, |delta| output.push_str(delta))?;
    let translated = clean(&output, text);
    if translated.is_empty() {
        return Err(OllamaError::Server("O modelo devolveu uma tradução vazia.".into()));
    }
    Ok(translated)
}
