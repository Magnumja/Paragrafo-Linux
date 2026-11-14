use crate::document::DocumentState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportFormat {
    Pdf,
    Markdown,
    Text,
}

impl ExportFormat {
    pub const ALL: [ExportFormat; 3] = [Self::Pdf, Self::Markdown, Self::Text];

    pub fn title(self) -> &'static str {
        match self {
            Self::Pdf => "PDF traduzido",
            Self::Markdown => "Markdown bilíngue",
            Self::Text => "Texto bilíngue",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Markdown => "md",
            Self::Text => "txt",
        }
    }
}

/// "relatorio.pdf" → "relatorio".
pub fn stem(filename: &str) -> &str {
    match filename.rfind('.') {
        Some(dot) if dot > 0 => &filename[..dot],
        _ => filename,
    }
}

pub fn text(filename: &str, target_name: &str, document: &DocumentState) -> String {
    let header = format!("{filename}\nTradução para {target_name} · {}/{} parágrafos\n\n", document.completed(), document.count());
    let body = document
        .paragraphs()
        .iter()
        .enumerate()
        .map(|(index, paragraph)| {
            let translated = if paragraph.translated.is_empty() { "[Não traduzido]" } else { &paragraph.translated };
            format!("Página {} · Parágrafo {}\n\nORIGINAL\n{}\n\nTRADUÇÃO\n{}", paragraph.page, index + 1, paragraph.original, translated)
        })
        .collect::<Vec<_>>()
        .join("\n\n————————————\n\n");
    header + &body
}

pub fn markdown(filename: &str, target_name: &str, document: &DocumentState) -> String {
    let mut out = format!("# {filename}\n\nTradução para {target_name} · {}/{} parágrafos\n", document.completed(), document.count());
    for &page in document.pages() {
        out += &format!("\n## Página {page}\n");
        for index in document.page_range(page).unwrap_or(0..0) {
            let paragraph = &document.paragraphs()[index];
            let translated = if paragraph.translated.is_empty() { "_[Não traduzido]_" } else { &paragraph.translated };
            out += &format!("\n{translated}\n\n");
            out += &paragraph.original.lines().filter(|line| !line.is_empty()).map(|line| format!("> {line}")).collect::<Vec<_>>().join("\n");
            out.push('\n');
        }
    }
    out
}
