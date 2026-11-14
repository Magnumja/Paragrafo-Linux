use crate::extraction::Paragraph;
use crate::ollama::OllamaMessage;
use crate::text::fold;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::Range;
use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    pub id: usize,
    pub page: usize,
    pub paragraphs: Range<usize>,
    pub text: String,
}

struct Draft {
    page: usize,
    start: usize,
    end: usize,
    text: String,
    length: usize,
    letters: bool,
}

fn length(text: &str) -> usize {
    text.chars().count()
}

pub const TARGET_LENGTH: usize = 900;
pub const MAX_LENGTH: usize = 1600;

/// Chunks with the lengths the search index uses.
pub fn chunks(paragraphs: &[Paragraph]) -> Vec<Chunk> {
    chunks_with(paragraphs, TARGET_LENGTH, MAX_LENGTH)
}

/// Groups consecutive paragraphs of the SAME page until ~target_length characters; never exceeds max_length unless a single
/// paragraph is longer, in which case that paragraph is split on sentence boundaries into several chunks that all carry the same one-paragraph range.
/// Paragraphs without any letters are attached to a neighbour, never a chunk on their own unless nothing else exists on the page.
/// Every paragraph index is covered by at least one chunk, in order.
pub fn chunks_with(paragraphs: &[Paragraph], target_length: usize, max_length: usize) -> Vec<Chunk> {
    let max_length = max_length.max(1);
    let target_length = target_length.clamp(1, max_length);
    let mut drafts: Vec<Draft> = Vec::new();
    let mut index = 0;
    while index < paragraphs.len() {
        let page = paragraphs[index].page;
        let page_start = drafts.len();
        let mut current: Option<Draft> = None;
        while index < paragraphs.len() && paragraphs[index].page == page {
            let at = index;
            index += 1;
            let text = paragraphs[at].original.trim().to_string();
            let size = length(&text);
            let letters = text.chars().any(char::is_alphabetic);
            if size > max_length {
                let mut pieces = split(&text, target_length, max_length);
                let mut start = at;
                if let Some(pending) = current.take() {
                    // A letterless run right before the long paragraph rides on its first piece when it fits.
                    if !pending.letters && pending.length + 1 + length(&pieces[0]) <= max_length {
                        pieces[0] = format!("{}\n{}", pending.text, pieces[0]);
                        start = pending.start;
                    } else {
                        drafts.push(pending);
                    }
                }
                for (offset, piece) in pieces.into_iter().enumerate() {
                    let size = length(&piece);
                    drafts.push(Draft { page, start: if offset == 0 { start } else { at }, end: at + 1, text: piece, length: size, letters: true });
                }
                continue;
            }
            let Some(mut open) = current.take() else {
                current = Some(Draft { page, start: at, end: at + 1, text, length: size, letters });
                continue;
            };
            let joined = open.length + usize::from(!open.text.is_empty() && !text.is_empty()) + size;
            if joined <= max_length && (open.length < target_length || !letters || !open.letters) {
                if !open.text.is_empty() && !text.is_empty() {
                    open.text.push('\n');
                }
                open.text.push_str(&text);
                open.length = joined;
                open.end = at + 1;
                open.letters = open.letters || letters;
                current = Some(open);
            } else {
                drafts.push(open);
                current = Some(Draft { page, start: at, end: at + 1, text, length: size, letters });
            }
        }
        if let Some(last) = current {
            // A trailing letterless run joins the previous chunk of this page when it fits.
            let fits = !last.letters && drafts.len() > page_start && drafts[drafts.len() - 1].length + 1 + last.length <= max_length;
            if fits {
                let previous = drafts.last_mut().expect("há um rascunho anterior");
                if !last.text.is_empty() {
                    previous.text.push('\n');
                    previous.text.push_str(&last.text);
                    previous.length += 1 + last.length;
                }
                previous.end = last.end;
            } else {
                drafts.push(last);
            }
        }
    }
    drafts.into_iter().enumerate().map(|(id, draft)| Chunk { id, page: draft.page, paragraphs: draft.start..draft.end, text: draft.text }).collect()
}

/// Packs sentences up to ~target_length; every piece is at most max_length characters.
fn split(text: &str, target_length: usize, max_length: usize) -> Vec<String> {
    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_length = 0;
    for sentence in sentences(text).into_iter().flat_map(|sentence| hard_split(&sentence, max_length)) {
        let size = length(&sentence);
        if current_length > 0 && current_length + 1 + size > target_length {
            pieces.push(std::mem::take(&mut current));
            current_length = 0;
        }
        if current_length > 0 {
            current.push(' ');
            current_length += 1;
        }
        current.push_str(&sentence);
        current_length += size;
    }
    if current_length > 0 {
        // Avoid a stub at the end when it fits in the previous piece.
        match pieces.last_mut() {
            Some(previous) if current_length < target_length / 4 && length(previous) + 1 + current_length <= max_length => {
                previous.push(' ');
                previous.push_str(&current);
            }
            _ => pieces.push(current),
        }
    }
    if pieces.is_empty() {
        vec![text.to_string()]
    } else {
        pieces
    }
}

const TERMINATORS: [char; 9] = ['.', '!', '?', '…', ';', '。', '！', '？', '；'];
const WIDE_TERMINATORS: [char; 4] = ['。', '！', '？', '；'];

fn sentences(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut previous: Option<char> = None;
    let mut flush = |current: &mut String| {
        let trimmed = current.trim();
        if !trimmed.is_empty() {
            result.push(trimmed.to_string());
        }
        current.clear();
    };
    for character in text.chars() {
        if let Some(previous) = previous {
            let boundary = character == '\n'
                || character == '\r'
                || (WIDE_TERMINATORS.contains(&previous) && !TERMINATORS.contains(&character))
                || (character.is_whitespace() && TERMINATORS.contains(&previous));
            if !current.is_empty() && boundary {
                flush(&mut current);
            }
        }
        current.push(character);
        previous = Some(character);
    }
    flush(&mut current);
    result
}

/// Breaks a sentence longer than max_length at word boundaries (or mid-word when there is no whitespace).
fn hard_split(sentence: &str, max_length: usize) -> Vec<String> {
    if length(sentence) <= max_length {
        return vec![sentence.to_string()];
    }
    let mut result = Vec::new();
    let mut current = String::new();
    let mut current_length = 0;
    for word in sentence.split_whitespace() {
        let mut word: Vec<char> = word.chars().collect();
        loop {
            let size = word.len();
            if current_length > 0 && current_length + 1 + size <= max_length {
                current.push(' ');
                current.extend(word.iter());
                current_length += 1 + size;
                break;
            }
            if current_length > 0 {
                result.push(std::mem::take(&mut current));
                current_length = 0;
            }
            if size <= max_length {
                current = word.iter().collect();
                current_length = size;
                break;
            }
            result.push(word[..max_length].iter().collect());
            word.drain(..max_length);
        }
    }
    if current_length > 0 {
        result.push(current);
    }
    result
}

#[derive(Clone, Debug, PartialEq)]
pub struct Retrieved {
    pub chunk: Chunk,
    pub score: f32,
}

pub struct VectorIndex {
    pub chunks: Vec<Chunk>,
    dimension: usize,
    /// chunks.len() rows × dimension, row-major. Rows without a usable vector stay zero and are only reachable lexically.
    matrix: Vec<f32>,
    /// Lexical corpus of each chunk's own text.
    term_counts: Vec<HashMap<String, usize>>,
    lengths: Vec<usize>,
}

pub const SEARCH_LIMIT: usize = 6;
pub const MINIMUM_SCORE: f32 = 0.25;

impl VectorIndex {
    pub fn new(chunks: Vec<Chunk>, vectors: &[Vec<f32>]) -> Self {
        let dimension = vectors.iter().find(|vector| !vector.is_empty()).map_or(0, Vec::len);
        let mut matrix = vec![0.0; chunks.len() * dimension];
        for row in 0..chunks.len().min(vectors.len()) {
            if dimension > 0 && vectors[row].len() == dimension {
                matrix[row * dimension..(row + 1) * dimension].copy_from_slice(&vectors[row]);
            }
        }
        let tokens: Vec<Vec<String>> = chunks.iter().map(|chunk| lexical_tokens(&chunk.text)).collect();
        let lengths = tokens.iter().map(Vec::len).collect();
        let term_counts = tokens.into_iter().map(counts).collect();
        Self { chunks, dimension, matrix, term_counts, lengths }
    }

    /// Hybrid retrieval: dense cosine + BM25 over tokens (lowercased, diacritic-folded, length >= 2, light stopword list for pt/en/es),
    /// fused with Reciprocal Rank Fusion (k=60). Lexical ranks only count for chunks with BM25 > 0. `extra_text[chunk id]`
    /// (e.g. the translation of the chunk) is appended to the lexical corpus for that chunk.
    /// Results sorted best-first, at most `limit`, dropping dense hits with cosine < minimum_score unless they also matched lexically.
    pub fn search(&self, query_vector: &[f32], query_text: &str, extra_text: &HashMap<usize, String>, limit: usize, minimum_score: f32) -> Vec<Retrieved> {
        let count = self.chunks.len();
        if count == 0 || limit == 0 {
            return Vec::new();
        }
        let cosine = self.cosines(query_vector);
        let lexical = self.bm25(query_text, extra_text);
        let descending = |values: &[f32], a: usize, b: usize| values[b].partial_cmp(&values[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b));

        // Plain RRF lets any chunk sharing one word with the question outrank the best semantic match, which hurts
        // most when the question and the document are in different languages. So: only the top of each list counts,
        // weak word matches are ignored, the lexical list weighs half, and the best semantic matches always keep a seat.
        let mut fused = vec![0.0f32; count];
        let mut candidate = vec![false; count];
        let (k, depth, lexical_weight) = (60.0f32, 30usize, 0.5f32);
        let mut pinned: Vec<usize> = Vec::new();
        if let Some(cosine) = &cosine {
            let mut order: Vec<usize> = (0..count).collect();
            order.sort_by(|&a, &b| descending(cosine, a, b));
            for (rank, &row) in order.iter().enumerate() {
                if cosine[row] < minimum_score {
                    continue;
                }
                candidate[row] = true;
                if rank < depth {
                    fused[row] += 1.0 / (k + (rank + 1) as f32);
                }
                if rank < (limit / 2).max(1) {
                    pinned.push(row);
                }
            }
        }
        let mut matched: Vec<usize> = (0..count).filter(|&row| lexical[row] > 0.0).collect();
        matched.sort_by(|&a, &b| descending(&lexical, a, b));
        let strongest = matched.first().map_or(0.0, |&row| lexical[row]);
        for (rank, &row) in matched.iter().take(depth).enumerate() {
            if lexical[row] >= 0.4 * strongest {
                fused[row] += lexical_weight / (k + (rank + 1) as f32);
                candidate[row] = true;
            }
        }

        let score = |row: usize| cosine.as_ref().map_or(0.0, |cosine| cosine[row]);
        let better = |a: &usize, b: &usize| {
            fused[*b]
                .partial_cmp(&fused[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(score(*b).partial_cmp(&score(*a)).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.cmp(b))
        };
        let mut chosen: Vec<usize> = (0..count).filter(|&row| candidate[row]).collect();
        chosen.sort_by(better);
        chosen.truncate(limit);
        for &row in &pinned {
            if !chosen.contains(&row) {
                if let Some(weakest) = chosen.iter().rposition(|row| !pinned.contains(row)) {
                    chosen[weakest] = row;
                }
            }
        }
        chosen.sort_by(better);
        chosen.into_iter().map(|row| Retrieved { chunk: self.chunks[row].clone(), score: score(row) }).collect()
    }

    /// Cosine of every row against the query, or None when there is no comparable query vector.
    fn cosines(&self, query: &[f32]) -> Option<Vec<f32>> {
        if self.dimension == 0 || query.len() != self.dimension || !query.iter().all(|value| value.is_finite()) {
            return None;
        }
        let norm = query.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm <= 0.0 {
            return None;
        }
        let unit: Vec<f32> = query.iter().map(|value| value / norm).collect();
        Some(self.matrix.chunks_exact(self.dimension).map(|row| row.iter().zip(&unit).map(|(a, b)| a * b).sum()).collect())
    }

    fn bm25(&self, query: &str, extra_text: &HashMap<usize, String>) -> Vec<f32> {
        let count = self.chunks.len();
        let mut scores = vec![0.0f32; count];
        let terms: BTreeSet<String> = lexical_tokens(query).into_iter().collect();
        if terms.is_empty() {
            return scores;
        }
        let mut extra_counts: HashMap<usize, HashMap<String, usize>> = HashMap::new();
        let mut lengths = self.lengths.clone();
        for (index, chunk) in self.chunks.iter().enumerate() {
            let Some(extra) = extra_text.get(&chunk.id).filter(|extra| !extra.is_empty()) else { continue };
            let tokens = lexical_tokens(extra);
            lengths[index] += tokens.len();
            extra_counts.insert(index, counts(tokens));
        }
        let average = (lengths.iter().sum::<usize>() as f32 / count as f32).max(1.0);
        let (k1, b) = (1.2f32, 0.75f32);
        for term in &terms {
            let frequencies: Vec<(usize, f32)> = (0..count)
                .filter_map(|row| {
                    let frequency = self.term_counts[row].get(term).copied().unwrap_or(0)
                        + extra_counts.get(&row).and_then(|counts| counts.get(term)).copied().unwrap_or(0);
                    (frequency > 0).then_some((row, frequency as f32))
                })
                .collect();
            if frequencies.is_empty() {
                continue;
            }
            let df = frequencies.len() as f32;
            let idf = (1.0 + (count as f32 - df + 0.5) / (df + 0.5)).ln();
            for (row, frequency) in frequencies {
                let normalization = k1 * (1.0 - b + b * lengths[row] as f32 / average);
                scores[row] += idf * frequency * (k1 + 1.0) / (frequency + normalization);
            }
        }
        scores
    }
}

fn lexical_tokens(text: &str) -> Vec<String> {
    let stopwords = stopwords();
    fold(text)
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.chars().count() >= 2 && !stopwords.contains(token))
        .map(str::to_string)
        .collect()
}

fn counts(tokens: Vec<String>) -> HashMap<String, usize> {
    let mut result = HashMap::new();
    for token in tokens {
        *result.entry(token).or_default() += 1;
    }
    result
}

/// Already diacritic-folded.
fn stopwords() -> &'static HashSet<&'static str> {
    static WORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    WORDS.get_or_init(|| {
        [
            // pt
            "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas", "um", "uma", "uns", "umas", "os", "as", "ao", "aos",
            "que", "qual", "quais", "quando", "onde", "como", "por", "para", "com", "sem", "se", "ou", "mas", "nao", "sim",
            "ser", "sao", "foi", "era", "esta", "estao", "este", "esse", "essa", "isso", "isto", "ele", "ela", "eles", "elas",
            "seu", "sua", "seus", "suas", "meu", "minha", "mais", "muito", "tambem", "ja", "sobre", "entre", "ate", "pelo", "pela",
            "tem", "ha", "quem", "porque",
            // en
            "the", "an", "and", "or", "of", "to", "in", "on", "at", "is", "are", "was", "were", "be", "been", "it", "its",
            "this", "that", "these", "those", "for", "with", "by", "from", "as", "what", "which", "who", "when", "where", "how",
            "why", "do", "does", "did", "not", "but", "if", "has", "have", "had", "can", "will", "about", "there", "their", "they",
            // es
            "el", "la", "los", "las", "un", "una", "unos", "unas", "del", "al", "en", "es", "son", "fue", "que", "cual", "cuales",
            "cuando", "donde", "por", "con", "sin", "su", "sus", "lo", "le", "les", "pero", "si", "mas", "muy", "tambien", "ya",
            "sobre", "entre", "hasta", "esta", "este", "esto", "ese", "esa", "hay", "quien", "porque",
        ]
        .into_iter()
        .collect()
    })
}

/// Answer ONLY from the numbered excerpts, in the language of the question, citing sources inline as [1], [2].
pub const SYSTEM_PROMPT: &str = "Você é o assistente de leitura do Parágrafo e responde a perguntas sobre um documento.
Regras:
1. Use SOMENTE as informações dos trechos numerados fornecidos na mensagem do usuário. Não use conhecimento externo.
2. Responda no mesmo idioma da pergunta, mesmo que os trechos estejam em outro idioma.
3. Cite as fontes no próprio texto, logo após a frase que elas sustentam, usando o número do trecho entre colchetes: [1], [2]. Toda afirmação tirada do documento precisa de citação.
4. Se os trechos não contiverem a resposta, diga claramente que o documento não traz essa informação. Não tente adivinhar.
5. Seja conciso: primeiro a resposta direta, depois, se necessário, um detalhe curto de apoio.
6. Nunca invente números de página, trechos ou citações. Cite apenas números de trechos que existem.";

/// [system] + up to the last 4 history messages + a user message containing the numbered excerpts
/// ("[1] (página 3)\n<text>") followed by the question.
pub fn messages(question: &str, sources: &[Retrieved], history: &[OllamaMessage]) -> Vec<OllamaMessage> {
    let question = question.trim();
    let mut content = if sources.is_empty() {
        "Trechos do documento: nenhum trecho relevante foi encontrado.\n\n".to_string()
    } else {
        let mut content = "Trechos do documento:\n\n".to_string();
        for (offset, source) in sources.iter().enumerate() {
            content += &format!("[{}] (página {})\n{}\n\n", offset + 1, source.chunk.page, source.chunk.text);
        }
        content
    };
    content += &format!("Pergunta: {question}\n\n");
    content += if sources.is_empty() {
        "Responda no idioma da pergunta dizendo que o documento não traz essa informação."
    } else {
        "Responda no idioma da pergunta, usando somente os trechos acima, e cite o número do trecho entre colchetes, por exemplo [1], logo após cada frase."
    };
    let mut result = vec![OllamaMessage::new("system", SYSTEM_PROMPT)];
    result.extend(history[history.len().saturating_sub(4)..].iter().cloned());
    result.push(OllamaMessage::new("user", content));
    result
}

/// 1-based source numbers actually cited in the answer text ("[2]", "[1, 3]", "[1][2]"), unique, ascending, only those in 1...source_count.
pub fn citations(answer: &str, source_count: usize) -> Vec<usize> {
    if source_count == 0 {
        return Vec::new();
    }
    let mut found = BTreeSet::new();
    let mut inside: Option<String> = None;
    for character in answer.chars() {
        if character == '[' {
            inside = Some(String::new());
        } else if character == ']' {
            if let Some(body) = inside.take() {
                found.extend(numbers(&body));
            }
        } else if let Some(body) = inside.as_mut() {
            // Only digits and list separators may appear inside a citation; anything else is ordinary bracketed text.
            if character.is_ascii_digit() || matches!(character, ',' | ';' | ' ' | '-') {
                body.push(character);
            } else {
                inside = None;
            }
        }
    }
    found.into_iter().filter(|number| (1..=source_count).contains(number)).collect()
}

fn numbers(body: &str) -> Vec<usize> {
    let mut result = Vec::new();
    for part in body.split([',', ';', ' ']).filter(|part| !part.is_empty()) {
        let bounds: Vec<Option<usize>> = part.split('-').map(|bound| bound.parse().ok()).collect();
        match bounds.as_slice() {
            [Some(value)] => result.push(*value),
            [Some(low), Some(high)] if low <= high && high - low <= 50 => result.extend(*low..=*high),
            _ => return Vec::new(),
        }
    }
    result
}
