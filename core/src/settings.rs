use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEFAULT_FONT_SIZE: f64 = 17.0;
pub const MIN_FONT_SIZE: f64 = 14.0;
pub const MAX_FONT_SIZE: f64 = 24.0;

/// Preferences kept between sessions in `$XDG_CONFIG_HOME/paragrafo/settings.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub font_size: f64,
    /// "parallel", "translation" or "original".
    pub reading_mode: String,
    pub show_assistant: bool,
    pub chat_model: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self { font_size: DEFAULT_FONT_SIZE, reading_mode: "parallel".into(), show_assistant: true, chat_model: crate::assistant::DEFAULT_CHAT_MODEL.into() }
    }
}

impl Settings {
    fn file() -> Option<PathBuf> {
        if let Some(path) = std::env::var_os("PARAGRAFO_SETTINGS") {
            return Some(PathBuf::from(path));
        }
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
            .map(|base| base.join("paragrafo/settings.json"))
    }

    pub fn load() -> Self {
        let mut settings: Settings = Self::file().and_then(|path| std::fs::read(path).ok()).and_then(|data| serde_json::from_slice(&data).ok()).unwrap_or_default();
        if !settings.font_size.is_finite() {
            settings.font_size = DEFAULT_FONT_SIZE;
        }
        settings.font_size = settings.font_size.clamp(MIN_FONT_SIZE, MAX_FONT_SIZE);
        settings
    }

    pub fn save(&self) {
        let (Some(path), Ok(data)) = (Self::file(), serde_json::to_vec_pretty(self)) else { return };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, data);
    }
}
