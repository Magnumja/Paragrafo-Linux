//! Turns a model answer into text whose citations ("[2]") and page mentions ("p. 12") are links back into the document.

use regex::{Captures, Regex};
use std::collections::HashSet;
use std::sync::OnceLock;

pub const SCHEME: &str = "paragrafo";

fn regex(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("regex válida"))
}

/// Markdown with internal links. `pages`, when given, limits page links to pages that really back the answer.
pub fn markdown(text: &str, source_count: usize, pages: Option<&HashSet<usize>>) -> String {
    static HEADING: OnceLock<Regex> = OnceLock::new();
    static BULLET: OnceLock<Regex> = OnceLock::new();
    static CITATION: OnceLock<Regex> = OnceLock::new();
    static PAGE: OnceLock<Regex> = OnceLock::new();
    let heading = regex(&HEADING, r"^\s*#{1,6}\s+(.+?)\s*#*\s*$");
    let bullet = regex(&BULLET, r"^(\s*)(?:[*\-•]\s+)+(.*)$");
    let citation = regex(&CITATION, r"\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]");
    let page = regex(&PAGE, r"(?i)\b(páginas?|paginas?|pág\.|pag\.|p\.|pages?)\s*(\d{1,4})\b");

    let lines: Vec<String> = text
        .split('\n')
        .map(|line| {
            if let Some(found) = heading.captures(line) {
                format!("**{}**", found[1].replace("**", ""))
            } else if let Some(found) = bullet.captures(line) {
                format!("{}• {}", &found[1], &found[2])
            } else {
                line.to_string()
            }
        })
        .collect();
    let out = lines.join("\n");
    let out = citation.replace_all(&out, |found: &Captures| {
        found[1]
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|number| number.parse::<usize>().ok())
            .filter(|number| source_count > 0 && (1..=source_count).contains(number))
            .map(|number| format!("[\\[{number}\\]]({SCHEME}://fonte/{number})"))
            .collect::<Vec<_>>()
            .join(" ")
    });
    let out = page.replace_all(&out, |found: &Captures| {
        match found[2].parse::<usize>() {
            Ok(number) if pages.is_none_or(|pages| pages.contains(&number)) => {
                format!("[{} {}]({SCHEME}://pagina/{number})", &found[1], &found[2])
            }
            _ => found[0].to_string(),
        }
    });
    out.into_owned()
}

#[derive(Clone, Debug, PartialEq)]
pub enum Segment {
    Text { text: String, bold: bool },
    Link { text: String, url: String },
}

/// Inline pieces of the markdown produced by `markdown()`: bold runs and links; everything else is literal.
pub fn segments(markdown: &str) -> Vec<Segment> {
    let chars: Vec<char> = markdown.chars().collect();
    let mut result: Vec<Segment> = Vec::new();
    let mut current = String::new();
    let mut bold = false;
    let mut at = 0;
    let flush = |current: &mut String, bold: bool, result: &mut Vec<Segment>| {
        if !current.is_empty() {
            result.push(Segment::Text { text: std::mem::take(current), bold });
        }
    };
    while at < chars.len() {
        let c = chars[at];
        if c == '\\' && at + 1 < chars.len() && "[]()*\\_".contains(chars[at + 1]) {
            current.push(chars[at + 1]);
            at += 2;
        } else if c == '*' && chars.get(at + 1) == Some(&'*') {
            flush(&mut current, bold, &mut result);
            bold = !bold;
            at += 2;
        } else if c == '[' {
            match link(&chars, at) {
                Some((text, url, next)) => {
                    flush(&mut current, bold, &mut result);
                    result.push(Segment::Link { text, url });
                    at = next;
                }
                None => {
                    current.push(c);
                    at += 1;
                }
            }
        } else {
            current.push(c);
            at += 1;
        }
    }
    flush(&mut current, bold, &mut result);
    result
}

/// `[text](url)` starting at `start`; the text may contain escaped brackets.
fn link(chars: &[char], start: usize) -> Option<(String, String, usize)> {
    let mut text = String::new();
    let mut at = start + 1;
    loop {
        match *chars.get(at)? {
            '\\' if at + 1 < chars.len() => {
                text.push(chars[at + 1]);
                at += 2;
            }
            ']' => break,
            '[' | '\n' => return None,
            c => {
                text.push(c);
                at += 1;
            }
        }
    }
    if chars.get(at + 1) != Some(&'(') {
        return None;
    }
    let close = chars[at + 2..].iter().position(|&c| c == ')' || c == '\n')? + at + 2;
    if chars[close] != ')' {
        return None;
    }
    let url: String = chars[at + 2..close].iter().collect();
    url.starts_with(&format!("{SCHEME}://")).then_some((text, url, close + 1))
}

/// The answer as the reader sees it, without markup.
pub fn plain(text: &str, source_count: usize, pages: Option<&HashSet<usize>>) -> String {
    segments(&markdown(text, source_count, pages))
        .into_iter()
        .map(|segment| match segment {
            Segment::Text { text, .. } | Segment::Link { text, .. } => text,
        })
        .collect()
}

pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Pango markup for a GTK label: bold runs and clickable internal links.
pub fn pango(text: &str, source_count: usize, pages: Option<&HashSet<usize>>) -> String {
    segments(&markdown(text, source_count, pages))
        .into_iter()
        .map(|segment| match segment {
            Segment::Text { text, bold: true } => format!("<b>{}</b>", escape(&text)),
            Segment::Text { text, bold: false } => escape(&text),
            Segment::Link { text, url } => format!("<a href=\"{}\">{}</a>", escape(&url), escape(&text)),
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Destination {
    Source(usize),
    Page(usize),
}

pub fn destination(url: &str) -> Option<Destination> {
    let rest = url.strip_prefix(SCHEME)?.strip_prefix("://")?;
    let (host, number) = rest.split_once('/')?;
    let number: usize = number.parse().ok().filter(|&n| n > 0)?;
    match host {
        "fonte" => Some(Destination::Source(number)),
        "pagina" => Some(Destination::Page(number)),
        _ => None,
    }
}
