use crate::text::sha256_hex;

/// A rectangle in PDF page coordinates (origin at the bottom-left, y rising).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self { x, y, width, height }
    }
    pub fn max_x(&self) -> f64 {
        self.x + self.width
    }
    pub fn max_y(&self) -> f64 {
        self.y + self.height
    }
    pub fn union(&self, other: &Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        Rect { x, y, width: self.max_x().max(other.max_x()) - x, height: self.max_y().max(other.max_y()) - y }
    }
}

pub fn union_all(rects: impl IntoIterator<Item = Rect>) -> Option<Rect> {
    rects.into_iter().fold(None, |area, rect| Some(area.map_or(rect, |a: Rect| a.union(&rect))))
}

#[derive(Clone, Debug, PartialEq)]
pub struct Paragraph {
    pub page: usize,
    pub original: String,
    pub translated: String,
    /// Union of the paragraph's lines in PDF page coordinates, used to highlight it in the original.
    pub bounds: Option<Rect>,
}

impl Paragraph {
    pub fn new(page: usize, original: impl Into<String>) -> Self {
        Self { page, original: original.into(), translated: String::new(), bounds: None }
    }
}

#[derive(Clone, Debug)]
pub struct TextLine {
    pub text: String,
    pub x: f64,
    pub y: f64,
    pub height: f64,
    pub width: f64,
}

impl TextLine {
    pub fn new(text: impl Into<String>, x: f64, y: f64, height: f64) -> Self {
        Self { text: text.into(), x, y, height, width: 0.0 }
    }
    pub fn wide(text: impl Into<String>, x: f64, y: f64, height: f64, width: f64) -> Self {
        Self { text: text.into(), x, y, height, width }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextBlock {
    pub text: String,
    pub bounds: Option<Rect>,
}

pub fn group(lines: &[TextLine]) -> Vec<String> {
    blocks(lines, false).into_iter().map(|block| block.text).collect()
}

/// PDF coordinates rise from the bottom. Preserves the extractor's reading order.
/// `split_on_size_change` separates headings from body text by line height. Only reliable for
/// embedded text; OCR line boxes vary with the letters on the line.
pub fn blocks(lines: &[TextLine], split_on_size_change: bool) -> Vec<TextBlock> {
    blocks_spaced(lines, split_on_size_change, None)
}

/// The usual distance between consecutive lines of a page (its lower quartile), or None when there are
/// too few lines to tell. A blank line between paragraphs is about twice this, whatever the font's box height.
pub fn typical_gap(lines: &[TextLine]) -> Option<f64> {
    let mut gaps: Vec<f64> = lines
        .windows(2)
        .map(|pair| (pair[0].y - pair[1].y, pair[0].height.max(pair[1].height)))
        .filter(|(gap, height)| *gap > 0.0 && *gap < height * 4.0)
        .map(|(gap, _)| gap)
        .collect();
    if gaps.len() < 3 {
        return None;
    }
    gaps.sort_by(|a, b| a.total_cmp(b));
    Some(gaps[gaps.len() / 4])
}

/// Like `blocks`, and also starts a paragraph where the gap is clearly larger than the page's usual line gap.
pub fn blocks_spaced(lines: &[TextLine], split_on_size_change: bool, typical_gap: Option<f64>) -> Vec<TextBlock> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut bounds: Option<Rect> = None;
    let mut previous: Option<&TextLine> = None;
    for line in lines {
        let text = line.text.trim();
        if text.is_empty() {
            continue;
        }
        if let Some(prev) = previous {
            let gap = prev.y - line.y;
            let boundary = gap > prev.height.max(line.height) * 1.65
                || typical_gap.is_some_and(|typical| gap > typical * 1.5)
                || gap < -prev.height * 0.6
                || (line.x - prev.x > line.height * 1.2 && gap > line.height * 0.5)
                || text.starts_with("• ")
                || (split_on_size_change && prev.height.max(line.height) > prev.height.min(line.height) * 1.25);
            if boundary && !current.is_empty() {
                result.push(TextBlock { text: std::mem::take(&mut current), bounds: bounds.take() });
            }
        }
        // A word hyphenated at the end of a line continues in lowercase on the next one.
        let mut tail = current.chars().rev();
        let hyphenated = tail.next() == Some('-')
            && tail.next().is_some_and(char::is_alphabetic)
            && text.chars().next().is_some_and(char::is_lowercase);
        if hyphenated {
            current.pop();
        } else if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(text);
        let rect = Rect::new(line.x, line.y, line.width, line.height);
        bounds = Some(bounds.map_or(rect, |area| area.union(&rect)));
        previous = Some(line);
    }
    if !current.is_empty() {
        result.push(TextBlock { text: current, bounds });
    }
    result
}

#[derive(Clone, Debug, PartialEq)]
pub enum ImportFailure {
    Invalid,
    Locked,
    Empty,
    Cancelled,
}

impl std::fmt::Display for ImportFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Invalid => "Não foi possível abrir este PDF.",
            Self::Locked => "Este PDF está protegido. Abra uma cópia desbloqueada.",
            Self::Empty => "Nenhum texto foi reconhecido neste PDF.",
            Self::Cancelled => "Importação cancelada.",
        })
    }
}

#[derive(Clone, Debug)]
pub struct Extracted {
    pub paragraphs: Vec<Paragraph>,
    pub ocr_pages: usize,
    pub empty_pages: Vec<usize>,
    pub page_count: usize,
    /// SHA-256 of the file, the key for saved translations and the search index.
    pub fingerprint: String,
    /// Pages had no text layer and no OCR engine was found to read them.
    pub ocr_missing: bool,
}

/// Identifies the extracted text the saved translations belong to.
pub fn content_key(paragraphs: &[Paragraph]) -> String {
    let joined = paragraphs.iter().map(|p| p.original.as_str()).collect::<Vec<_>>().join("\u{1F}");
    sha256_hex(joined.as_bytes())
}
