use crate::text::now;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// What is kept between sessions for one PDF, keyed by the file's fingerprint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SavedDocument {
    pub name: String,
    pub path: String,
    pub pages: usize,
    pub source: String,
    pub target: String,
    pub translations: Vec<String>,
    /// Hash of the extracted paragraphs the translations belong to.
    #[serde(default)]
    pub content_key: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub summary_sources: Option<Vec<SavedSource>>,
    /// Identifies what the saved vectors were computed from: model, chunker version and the chunk texts.
    #[serde(default)]
    pub index_key: Option<String>,
    #[serde(default)]
    pub dimension: Option<usize>,
    /// Seconds since the Unix epoch.
    #[serde(default = "now")]
    pub opened: f64,
}

impl SavedDocument {
    pub fn new(name: &str, path: &str, pages: usize, source: &str, target: &str, translations: Vec<String>) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            pages,
            source: source.into(),
            target: target.into(),
            translations,
            content_key: None,
            summary: None,
            summary_sources: None,
            index_key: None,
            dimension: None,
            opened: now(),
        }
    }
}

/// A passage a saved synthesis points to.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedSource {
    pub page: usize,
    pub paragraph: usize,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecentDocument {
    pub id: String,
    pub name: String,
    pub path: String,
    pub pages: usize,
    pub paragraphs: usize,
    pub completed: usize,
    pub opened: f64,
}

/// Plain files in the user's data folder. Translations as JSON, embeddings as raw little-endian Float32.
#[derive(Clone, Debug)]
pub struct Library {
    directory: PathBuf,
}

impl Default for Library {
    /// `$PARAGRAFO_LIBRARY` (used by checks so they never touch real saved work),
    /// else `$XDG_DATA_HOME/paragrafo`, else `~/.local/share/paragrafo`.
    fn default() -> Self {
        let directory = std::env::var_os("PARAGRAFO_LIBRARY").map(PathBuf::from).unwrap_or_else(|| {
            std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
                .unwrap_or_else(std::env::temp_dir)
                .join("paragrafo")
        });
        Self::at(directory)
    }
}

impl Library {
    pub fn at(directory: impl Into<PathBuf>) -> Self {
        let directory = directory.into();
        if fs::create_dir_all(&directory).is_ok() {
            let _ = fs::set_permissions(&directory, fs::Permissions::from_mode(0o700));
        }
        Self { directory }
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Only names made of letters, digits, "-" and "_" map to a file, and never through a symlink:
    /// a derived path must stay inside the library.
    fn file(&self, fingerprint: &str, extension: &str) -> Option<PathBuf> {
        let valid = !fingerprint.is_empty()
            && fingerprint.len() <= 128
            && fingerprint.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
        if !valid {
            return None;
        }
        let path = self.directory.join(format!("{fingerprint}.{extension}"));
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => None,
            _ => Some(path),
        }
    }

    fn recents_file(&self) -> PathBuf {
        self.directory.join("recentes.json")
    }

    /// Atomic: a crash never leaves half a file behind. Readable by the owner only.
    fn write(&self, data: &[u8], path: &Path) {
        let temporary = path.with_extension(format!("tmp{}", std::process::id()));
        let written = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&temporary)
            .and_then(|mut file| file.write_all(data).and_then(|_| file.sync_all()));
        if written.is_ok() && fs::rename(&temporary, path).is_ok() {
            return;
        }
        let _ = fs::remove_file(&temporary);
    }

    pub fn load(&self, fingerprint: &str) -> Option<SavedDocument> {
        let data = fs::read(self.file(fingerprint, "json")?).ok()?;
        serde_json::from_slice(&data).ok()
    }

    pub fn save(&self, document: &SavedDocument, fingerprint: &str) {
        let (Some(path), Ok(data)) = (self.file(fingerprint, "json"), serde_json::to_vec(document)) else { return };
        self.write(&data, &path);
        let mut list: Vec<RecentDocument> = self.recents().into_iter().filter(|recent| recent.id != fingerprint).collect();
        list.insert(
            0,
            RecentDocument {
                id: fingerprint.into(),
                name: document.name.clone(),
                path: document.path.clone(),
                pages: document.pages,
                paragraphs: document.translations.len(),
                completed: document.translations.iter().filter(|text| !text.is_empty()).count(),
                opened: document.opened,
            },
        );
        // The list is trimmed; saved work is never deleted behind the user's back.
        list.truncate(50);
        if let Ok(encoded) = serde_json::to_vec(&list) {
            self.write(&encoded, &self.recents_file());
        }
    }

    /// Read-modify-write for callers that own only part of the record.
    pub fn update(&self, fingerprint: &str, change: impl FnOnce(&mut SavedDocument)) {
        if let Some(mut document) = self.load(fingerprint) {
            change(&mut document);
            self.save(&document, fingerprint);
        }
    }

    pub fn recents(&self) -> Vec<RecentDocument> {
        fs::read(self.recents_file()).ok().and_then(|data| serde_json::from_slice(&data).ok()).unwrap_or_default()
    }

    pub fn remove(&self, fingerprint: &str) {
        let (Some(json), Some(vectors)) = (self.file(fingerprint, "json"), self.file(fingerprint, "vec")) else { return };
        let _ = fs::remove_file(json);
        let _ = fs::remove_file(vectors);
        let list: Vec<RecentDocument> = self.recents().into_iter().filter(|recent| recent.id != fingerprint).collect();
        if let Ok(encoded) = serde_json::to_vec(&list) {
            self.write(&encoded, &self.recents_file());
        }
    }

    pub fn save_vectors(&self, vectors: &[Vec<f32>], fingerprint: &str) {
        let Some(path) = self.file(fingerprint, "vec") else { return };
        let mut data = Vec::with_capacity(vectors.iter().map(Vec::len).sum::<usize>() * 4);
        for value in vectors.iter().flatten() {
            data.extend_from_slice(&value.to_le_bytes());
        }
        self.write(&data, &path);
    }

    pub fn load_vectors(&self, fingerprint: &str, count: usize, dimension: usize) -> Option<Vec<Vec<f32>>> {
        const STRIDE: usize = 4;
        if count == 0 || count > 100_000 || dimension == 0 || dimension > 8192 || count > 128 * 1024 * 1024 / STRIDE / dimension {
            return None;
        }
        let data = fs::read(self.file(fingerprint, "vec")?).ok()?;
        if data.len() != count * dimension * STRIDE {
            return None;
        }
        Some(
            data.chunks_exact(dimension * STRIDE)
                .map(|row| row.as_chunks::<STRIDE>().0.iter().map(|bytes| f32::from_le_bytes(*bytes)).collect())
                .collect(),
        )
    }
}
