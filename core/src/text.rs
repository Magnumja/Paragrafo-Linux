use sha2::{Digest, Sha256};
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};

/// Lowercase, without diacritics and with compatibility forms unified ("Tradução" → "traducao").
pub fn fold(text: &str) -> String {
    text.nfkd().filter(|c| !is_combining_mark(*c)).flat_map(char::to_lowercase).collect()
}

/// The first `count` characters, never cutting one in half.
pub fn prefix(text: &str, count: usize) -> &str {
    match text.char_indices().nth(count) {
        Some((end, _)) => &text[..end],
        None => text,
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Seconds since the Unix epoch.
pub fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// "1 página", "3 páginas".
pub fn plural(count: usize, one: &str, many: &str) -> String {
    format!("{count} {}", if count == 1 { one } else { many })
}
