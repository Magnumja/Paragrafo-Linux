use crate::extraction::Paragraph;
use crate::text::prefix;

pub struct Language {
    /// ISO 639-1, what is saved and shown.
    pub code: &'static str,
    /// ISO 639-3, what the language detector reports.
    pub iso3: &'static str,
    pub name: &'static str,
    /// English name, used in the translation prompt.
    pub english: &'static str,
}

const fn language(code: &'static str, iso3: &'static str, name: &'static str, english: &'static str) -> Language {
    Language { code, iso3, name, english }
}

/// The most used languages first, then the rest in Portuguese alphabetical order.
pub const LANGUAGES: &[Language] = &[
    language("pt", "por", "Português", "Portuguese"),
    language("en", "eng", "Inglês", "English"),
    language("es", "spa", "Espanhol", "Spanish"),
    language("de", "deu", "Alemão", "German"),
    language("ar", "ara", "Árabe", "Arabic"),
    language("bn", "ben", "Bengali", "Bengali"),
    language("bg", "bul", "Búlgaro", "Bulgarian"),
    language("ca", "cat", "Catalão", "Catalan"),
    language("zh", "cmn", "Chinês", "Chinese"),
    language("ko", "kor", "Coreano", "Korean"),
    language("hr", "hrv", "Croata", "Croatian"),
    language("da", "dan", "Dinamarquês", "Danish"),
    language("sk", "slk", "Eslovaco", "Slovak"),
    language("fi", "fin", "Finlandês", "Finnish"),
    language("fr", "fra", "Francês", "French"),
    language("el", "ell", "Grego", "Greek"),
    language("he", "heb", "Hebraico", "Hebrew"),
    language("hi", "hin", "Hindi", "Hindi"),
    language("nl", "nld", "Holandês", "Dutch"),
    language("hu", "hun", "Húngaro", "Hungarian"),
    language("id", "ind", "Indonésio", "Indonesian"),
    language("it", "ita", "Italiano", "Italian"),
    language("ja", "jpn", "Japonês", "Japanese"),
    language("nb", "nob", "Norueguês", "Norwegian"),
    language("fa", "pes", "Persa", "Persian"),
    language("pl", "pol", "Polonês", "Polish"),
    language("ro", "ron", "Romeno", "Romanian"),
    language("ru", "rus", "Russo", "Russian"),
    language("sr", "srp", "Sérvio", "Serbian"),
    language("sv", "swe", "Sueco", "Swedish"),
    language("th", "tha", "Tailandês", "Thai"),
    language("cs", "ces", "Tcheco", "Czech"),
    language("tr", "tur", "Turco", "Turkish"),
    language("uk", "ukr", "Ucraniano", "Ukrainian"),
    language("vi", "vie", "Vietnamita", "Vietnamese"),
];

fn base(code: &str) -> String {
    code.split(['-', '_']).next().unwrap_or("").to_lowercase()
}

pub fn find(code: &str) -> Option<&'static Language> {
    let wanted = base(code);
    LANGUAGES.iter().find(|language| language.code == wanted)
}

/// Portuguese name of a language code; the code itself when it is not known.
pub fn language_name(code: &str) -> String {
    find(code).map_or_else(|| code.to_string(), |language| language.name.to_string())
}

pub fn english_name(code: &str) -> String {
    find(code).map_or_else(|| code.to_string(), |language| language.english.to_string())
}

pub fn needs_translation(text: &str) -> bool {
    text.chars().any(char::is_alphabetic)
}

/// Dominant language of the document, or None when there is no text to judge
/// (numbers only) or the language is not one the app can name.
pub fn detect_language(paragraphs: &[Paragraph]) -> Option<String> {
    let mut sample = String::new();
    let mut length = 0;
    for paragraph in paragraphs.iter().filter(|p| needs_translation(&p.original)) {
        let piece = prefix(&paragraph.original, 2000);
        sample.push_str(piece);
        sample.push('\n');
        length += piece.chars().count() + 1;
        if length >= 12000 {
            break;
        }
    }
    if sample.is_empty() {
        return None;
    }
    // whatlang knows more languages and scripts; whichlang is more accurate for the sixteen it covers.
    let broad = whatlang::detect(&sample);
    let precise = whichlang::detect_language(&sample).three_letter_code();
    const PRECISE: [&str; 16] = ["ara", "cmn", "deu", "eng", "fra", "hin", "ita", "jpn", "kor", "nld", "por", "rus", "spa", "swe", "tur", "vie"];
    let detected = match broad {
        Some(info) if info.is_reliable() && !PRECISE.contains(&info.lang().code()) => info.lang().code(),
        _ => precise,
    };
    LANGUAGES.iter().find(|language| language.iso3 == detected).map(|language| language.code.to_string())
}

/// "pt" and "pt-BR" are the same language for translation purposes.
pub fn same_language(source: &str, target: &str) -> bool {
    let (a, b) = (base(source), base(target));
    !a.is_empty() && a == b
}
