//! Everything that touches a PDF: text extraction with Poppler, OCR with Tesseract for scanned pages,
//! on-screen rendering with the marked passage, and the translated PDF export.

use adw::prelude::*;
use gtk::glib;
use paragrafo_core::document::DocumentState;
use paragrafo_core::exporter;
use paragrafo_core::extraction::{blocks_spaced, typical_gap, Extracted, ImportFailure, Paragraph, Rect, TextLine};
use paragrafo_core::text::sha256_hex;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

/// Files larger than this are refused instead of being read whole into memory.
const MAX_FILE_BYTES: u64 = 1024 * 1024 * 1024;

pub fn open(path: &Path) -> Option<poppler::Document> {
    let uri = glib::filename_to_uri(path, None).ok()?;
    poppler::Document::from_file(&uri, None).ok()
}

pub fn extract(path: &Path, cancel: &AtomicBool, mut progress: impl FnMut(usize, usize)) -> Result<Extracted, ImportFailure> {
    if std::fs::metadata(path).map_err(|_| ImportFailure::Invalid)?.len() > MAX_FILE_BYTES {
        return Err(ImportFailure::Invalid);
    }
    let data = std::fs::read(path).map_err(|_| ImportFailure::Invalid)?;
    let fingerprint = sha256_hex(&data);
    let document = poppler::Document::from_bytes(&glib::Bytes::from_owned(data), None)
        .map_err(|error| if error.matches(poppler::Error::Encrypted) { ImportFailure::Locked } else { ImportFailure::Invalid })?;
    let page_count = document.n_pages().max(0) as usize;
    let mut paragraphs = Vec::new();
    let mut ocr_pages = 0;
    let mut ocr_missing = false;
    let mut empty_pages = Vec::new();
    for index in 0..page_count {
        if cancel.load(Ordering::Relaxed) {
            return Err(ImportFailure::Cancelled);
        }
        progress(index + 1, page_count);
        let Some(page) = document.page(index as i32) else {
            empty_pages.push(index + 1);
            continue;
        };
        let (width, height) = page.size();
        let mut lines = text_lines(&page, height);
        let recognized = lines.is_empty();
        if recognized {
            match ocr(&page, width, height) {
                Some(found) => {
                    ocr_pages += 1;
                    lines = found;
                }
                None => ocr_missing = true,
            }
        }
        if std::env::var_os("PARAGRAFO_DEBUG_LINES").is_some() {
            for line in &lines {
                eprintln!("p{} y={:.1} h={:.1} x={:.1} {:?}", index + 1, line.y, line.height, line.x, line.text);
            }
        }
        // OCR boxes are too irregular for the line-gap rule; it is for the text layer only.
        let found = blocks_spaced(&lines, !recognized, if recognized { None } else { typical_gap(&lines) });
        if found.is_empty() {
            empty_pages.push(index + 1);
        }
        paragraphs.extend(found.into_iter().map(|block| Paragraph { page: index + 1, original: block.text, translated: String::new(), bounds: block.bounds }));
    }
    if cancel.load(Ordering::Relaxed) {
        return Err(ImportFailure::Cancelled);
    }
    if paragraphs.is_empty() {
        return Err(ImportFailure::Empty);
    }
    Ok(Extracted { paragraphs, ocr_pages, empty_pages, page_count, fingerprint, ocr_missing })
}

/// Lines of the page's text layer with their boxes, in PDF coordinates (y rising from the bottom).
fn text_lines(page: &poppler::Page, height: f64) -> Vec<TextLine> {
    let Some(text) = page.text() else { return Vec::new() };
    let layout = text_layout(page);
    let mut lines = Vec::new();
    let mut current = String::new();
    // left, top, right, bottom with y falling, as Poppler reports it.
    let mut area: Option<(f64, f64, f64, f64)> = None;
    let mut flush = |current: &mut String, area: &mut Option<(f64, f64, f64, f64)>| {
        if let (false, Some((left, top, right, bottom))) = (current.trim().is_empty(), *area) {
            lines.push(TextLine::wide(current.as_str(), left, height - bottom, bottom - top, right - left));
        }
        current.clear();
        *area = None;
    };
    // One rectangle per character of the text, line breaks included.
    let mut rectangles = layout.iter();
    for character in text.chars() {
        let rectangle = rectangles.next();
        if character == '\n' {
            flush(&mut current, &mut area);
            continue;
        }
        current.push(character);
        if let (false, Some(&(x1, y1, x2, y2))) = (character.is_whitespace(), rectangle) {
            let (left, right) = (x1.min(x2), x1.max(x2));
            let (top, bottom) = (y1.min(y2), y1.max(y2));
            area = Some(area.map_or((left, top, right, bottom), |a| (a.0.min(left), a.1.min(top), a.2.max(right), a.3.max(bottom))));
        }
    }
    flush(&mut current, &mut area);
    lines
}

/// The box of every character of `page.text()`, as (x1, y1, x2, y2) with y falling from the top.
/// The safe binding does not expose this call.
fn text_layout(page: &poppler::Page) -> Vec<(f64, f64, f64, f64)> {
    use glib::translate::ToGlibPtr;
    let mut rectangles: *mut poppler_sys::PopplerRectangle = std::ptr::null_mut();
    let mut count: std::ffi::c_uint = 0;
    // SAFETY: Poppler fills `rectangles` with `count` items that we own and must release with g_free.
    unsafe {
        if poppler_sys::poppler_page_get_text_layout(page.to_glib_none().0, &mut rectangles, &mut count) == 0 || rectangles.is_null() {
            return Vec::new();
        }
        let result = std::slice::from_raw_parts(rectangles, count as usize).iter().map(|r| (r.x1, r.y1, r.x2, r.y2)).collect();
        glib::ffi::g_free(rectangles.cast());
        result
    }
}

/// Languages handed to Tesseract, or None when it is not installed.
fn ocr_languages() -> Option<&'static str> {
    static LANGUAGES: OnceLock<Option<String>> = OnceLock::new();
    LANGUAGES
        .get_or_init(|| {
            let output = Command::new("tesseract").arg("--list-langs").stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
            let installed: Vec<String> =
                String::from_utf8_lossy(&output.stdout).lines().skip(1).map(|line| line.trim().to_string()).filter(|name| !name.is_empty() && name != "osd").collect();
            let valid = |name: &str| !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if let Ok(wanted) = std::env::var("PARAGRAFO_OCR_LANGS") {
                let chosen: Vec<&str> = wanted.split('+').filter(|name| valid(name) && installed.iter().any(|i| i == name)).collect();
                if !chosen.is_empty() {
                    return Some(chosen.join("+"));
                }
            }
            let preferred: Vec<&str> = ["por", "eng", "spa", "fra", "deu", "ita"].into_iter().filter(|name| installed.iter().any(|i| i == name)).collect();
            if !preferred.is_empty() {
                return Some(preferred.join("+"));
            }
            installed.into_iter().find(|name| valid(name))
        })
        .as_deref()
}

/// Recognizes the text of a page without a text layer. None when no OCR engine is available.
fn ocr(page: &poppler::Page, width: f64, height: f64) -> Option<Vec<TextLine>> {
    let languages = ocr_languages()?;
    let scale = (2400.0 / width.max(height).max(1.0)).min(3.0);
    let surface = cairo::ImageSurface::create(cairo::Format::Rgb24, (width * scale).ceil() as i32, (height * scale).ceil() as i32).ok()?;
    {
        let context = cairo::Context::new(&surface).ok()?;
        context.set_source_rgb(1.0, 1.0, 1.0);
        context.paint().ok()?;
        context.scale(scale, scale);
        page.render(&context);
    }
    let mut png = Vec::new();
    surface.write_to_png(&mut png).ok()?;
    let mut child = Command::new("tesseract")
        .args(["stdin", "stdout", "-l", languages, "--psm", "3", "tsv"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&png);
    });
    let output = child.wait_with_output().ok()?;
    let _ = writer.join();
    if !output.status.success() {
        return Some(Vec::new());
    }
    Some(ocr_lines(&String::from_utf8_lossy(&output.stdout), scale, height))
}

/// Tesseract's TSV (one word per row) grouped into lines, mapped to the page so highlights line up.
fn ocr_lines(tsv: &str, scale: f64, height: f64) -> Vec<TextLine> {
    struct Line {
        key: (u32, u32, u32),
        text: String,
        left: f64,
        top: f64,
        right: f64,
        bottom: f64,
    }
    let mut lines: Vec<Line> = Vec::new();
    for row in tsv.lines().skip(1) {
        let fields: Vec<&str> = row.split('\t').collect();
        if fields.len() < 12 || fields[0] != "5" {
            continue;
        }
        let number = |at: usize| fields[at].parse::<f64>().unwrap_or(0.0);
        let word = fields[11].trim();
        if word.is_empty() || number(10) < 0.0 {
            continue;
        }
        let key = (number(2) as u32, number(3) as u32, number(4) as u32);
        let (left, top) = (number(6), number(7));
        let (right, bottom) = (left + number(8), top + number(9));
        match lines.last_mut() {
            Some(line) if line.key == key => {
                line.text.push(' ');
                line.text.push_str(word);
                line.left = line.left.min(left);
                line.top = line.top.min(top);
                line.right = line.right.max(right);
                line.bottom = line.bottom.max(bottom);
            }
            _ => lines.push(Line { key, text: word.to_string(), left, top, right, bottom }),
        }
    }
    lines
        .into_iter()
        .map(|line| TextLine::wide(line.text, line.left / scale, height - line.bottom / scale, (line.bottom - line.top) / scale, (line.right - line.left) / scale))
        .collect()
}

/// One page of the original, fitted to the available width, with the passage marked and scrolled into view.
pub fn view(document: &poppler::Document, page: usize, highlight: Option<Rect>) -> gtk::ScrolledWindow {
    let scroll = gtk::ScrolledWindow::builder().hscrollbar_policy(gtk::PolicyType::Never).vexpand(true).hexpand(true).build();
    scroll.add_css_class("pdf-backdrop");
    let Some(pdf_page) = document.page(page as i32 - 1) else { return scroll };
    let (page_width, page_height) = pdf_page.size();
    let area = gtk::DrawingArea::builder().hexpand(true).margin_top(16).margin_bottom(16).margin_start(16).margin_end(16).build();
    area.set_content_height(page_height as i32);
    area.set_accessible_role(gtk::AccessibleRole::Img);
    area.update_property(&[gtk::accessible::Property::Label(&format!("Página {page} do PDF original"))]);
    area.set_draw_func(move |_, context, width, _| {
        let scale = f64::from(width) / page_width.max(1.0);
        context.set_source_rgb(1.0, 1.0, 1.0);
        context.rectangle(0.0, 0.0, page_width * scale, page_height * scale);
        let _ = context.fill();
        let _ = context.save();
        context.scale(scale, scale);
        pdf_page.render(context);
        if let Some(mark) = highlight {
            context.set_source_rgba(0.45, 0.80, 0.62, 0.42);
            context.rectangle(mark.x - 3.0, page_height - mark.max_y() - 2.0, mark.width + 6.0, mark.height + 4.0);
            let _ = context.fill();
        }
        let _ = context.restore();
    });
    area.connect_resize(move |area, width, _| {
        let wanted = (f64::from(width) * page_height / page_width.max(1.0)).ceil() as i32;
        if area.content_height() != wanted {
            area.set_content_height(wanted);
        }
    });
    let clamp = adw::Clamp::builder().maximum_size(980).child(&area).build();
    scroll.set_child(Some(&clamp));
    if let Some(mark) = highlight {
        let (scroll, area) = (scroll.clone(), area.clone());
        glib::timeout_add_local_once(Duration::from_millis(220), move || {
            let scale = f64::from(area.width()) / page_width.max(1.0);
            scroll.vadjustment().set_value(((page_height - mark.max_y()) * scale - 120.0).max(0.0));
        });
    }
    scroll
}

const PAGE_WIDTH: f64 = 595.0;
const PAGE_HEIGHT: f64 = 842.0;
const MARGIN_X: f64 = 76.0;
const MARGIN_Y: f64 = 78.0;
const INK: (f64, f64, f64) = (0.1, 0.1, 0.1);
const MUTED: (f64, f64, f64) = (0.45, 0.45, 0.45);

struct PdfWriter {
    context: cairo::Context,
    layout: pango::Layout,
    y: f64,
    number: usize,
}

impl PdfWriter {
    fn footer(&self) {
        let footer = pangocairo::functions::create_layout(&self.context);
        pangocairo::functions::context_set_resolution(&footer.context(), 72.0);
        footer.set_font_description(Some(&pango::FontDescription::from_string("sans 8.5")));
        footer.set_text(&self.number.to_string());
        let (width, _) = footer.pixel_size();
        self.context.set_source_rgb(MUTED.0, MUTED.1, MUTED.2);
        self.context.move_to(PAGE_WIDTH / 2.0 - f64::from(width) / 2.0, PAGE_HEIGHT - 50.0);
        pangocairo::functions::show_layout(&self.context, &footer);
    }

    /// Lays a block of text out line by line, starting a new page whenever the next line does not fit.
    fn block(&mut self, text: &str, font: &str, color: (f64, f64, f64), before: f64, after: f64, line_spacing: f64) {
        self.layout.set_font_description(Some(&pango::FontDescription::from_string(font)));
        self.layout.set_text(text);
        self.y += before;
        for index in 0..self.layout.line_count() {
            let Some(line) = self.layout.line_readonly(index) else { continue };
            let (_, logical) = line.extents();
            let height = f64::from(logical.height()) / f64::from(pango::SCALE);
            let ascent = -f64::from(logical.y()) / f64::from(pango::SCALE);
            if self.y + height > PAGE_HEIGHT - MARGIN_Y {
                self.footer();
                let _ = self.context.show_page();
                self.number += 1;
                self.y = MARGIN_Y;
            }
            self.context.set_source_rgb(color.0, color.1, color.2);
            self.context.move_to(MARGIN_X, self.y + ascent);
            pangocairo::functions::show_layout_line(&self.context, &line);
            self.y += height + line_spacing;
        }
        self.y += after;
    }
}

/// A clean reading copy of the translation on A4. Untranslated passages stay in the original, in gray.
pub fn export(path: &Path, filename: &str, target_name: &str, document: &DocumentState) -> Result<(), String> {
    let failed = |error: cairo::Error| format!("Não foi possível gerar o PDF: {error}");
    let surface = cairo::PdfSurface::new(PAGE_WIDTH, PAGE_HEIGHT, path).map_err(failed)?;
    let context = cairo::Context::new(&surface).map_err(failed)?;
    let layout = pangocairo::functions::create_layout(&context);
    // One unit is one point, so font sizes mean what they say on paper.
    pangocairo::functions::context_set_resolution(&layout.context(), 72.0);
    layout.set_width(((PAGE_WIDTH - 2.0 * MARGIN_X) * f64::from(pango::SCALE)) as i32);
    layout.set_wrap(pango::WrapMode::WordChar);
    let mut writer = PdfWriter { context, layout, y: MARGIN_Y, number: 1 };
    writer.block(exporter::stem(filename), "serif Semi-Bold 22", INK, 0.0, 8.0, 2.0);
    writer.block(&format!("Tradução para {target_name} · {} de {} parágrafos", document.completed(), document.count()), "sans 9.5", MUTED, 0.0, 8.0, 2.0);
    for &page in document.pages() {
        writer.block(&format!("PÁGINA {page}"), "sans Semi-Bold 8", MUTED, 14.0, 8.0, 2.0);
        for index in document.page_range(page).unwrap_or(0..0) {
            let paragraph = &document.paragraphs()[index];
            let pending = paragraph.translated.is_empty();
            writer.block(if pending { &paragraph.original } else { &paragraph.translated }, "serif 11.5", if pending { MUTED } else { INK }, 0.0, 11.0, 4.0);
        }
    }
    writer.footer();
    let _ = writer.context.show_page();
    drop(writer);
    surface.finish();
    surface.status().map_err(failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temporary(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("paragrafo-{}-{name}", std::process::id()))
    }

    fn text_of(path: &Path) -> (usize, String) {
        let document = open(path).expect("PDF gerado abre");
        let pages = document.n_pages() as usize;
        let text = (0..pages).filter_map(|index| document.page(index as i32)?.text()).map(|text| text.to_string()).collect::<Vec<_>>().join("\n");
        (pages, text)
    }

    fn written(path: &Path, draw: impl Fn(&cairo::Context)) {
        let surface = cairo::PdfSurface::new(500.0, 700.0, path).unwrap();
        let context = cairo::Context::new(&surface).unwrap();
        draw(&context);
        context.show_page().unwrap();
        drop(context);
        surface.finish();
    }

    #[test]
    fn extracao_de_pdf_textual() {
        let path = temporary("texto.pdf");
        written(&path, |context| {
            context.select_font_face("sans", cairo::FontSlant::Normal, cairo::FontWeight::Normal);
            context.set_font_size(16.0);
            context.move_to(40.0, 100.0);
            context.show_text("Hello world. Revenue: 1250.").unwrap();
            context.move_to(40.0, 160.0);
            context.show_text("Second paragraph.").unwrap();
        });
        let go = AtomicBool::new(false);
        let mut seen = Vec::new();
        let extracted = extract(&path, &go, |page, total| seen.push((page, total))).expect("extração");
        assert_eq!(extracted.paragraphs.len(), 2, "extração de PDF real com dois parágrafos");
        assert!(extracted.paragraphs[0].original.contains("1250"), "números preservados no PDF");
        assert_eq!(extracted.ocr_pages, 0, "PDF textual dispensa OCR");
        assert!(extracted.paragraphs.iter().all(|p| p.bounds.is_some_and(|b| b.width > 0.0)) && extracted.page_count == 1 && extracted.fingerprint.len() == 64, "extração traz área, contagem de páginas e impressão digital");
        let first = extracted.paragraphs[0].bounds.unwrap();
        let second = extracted.paragraphs[1].bounds.unwrap();
        assert!(first.y > second.y && (first.x - 40.0).abs() < 4.0 && first.max_y() < 700.0 - 80.0 && first.y > 700.0 - 110.0, "área em coordenadas do PDF, com y crescendo para cima");
        assert_eq!(seen, [(1, 1)], "progresso por página");

        let stopped = AtomicBool::new(true);
        assert_eq!(extract(&path, &stopped, |_, _| {}).unwrap_err(), ImportFailure::Cancelled, "cancelamento da tarefa de importação");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn arquivos_invalidos() {
        let go = AtomicBool::new(false);
        let path = temporary("lixo.pdf");
        std::fs::write(&path, b"isto nao e um pdf").unwrap();
        assert_eq!(extract(&path, &go, |_, _| {}).unwrap_err(), ImportFailure::Invalid, "arquivo que não é PDF é recusado");
        std::fs::remove_file(&path).unwrap();
        assert_eq!(extract(Path::new("/nao/existe.pdf"), &go, |_, _| {}).unwrap_err(), ImportFailure::Invalid, "arquivo ausente é recusado");
        let blank = temporary("vazio.pdf");
        written(&blank, |_| {});
        let result = extract(&blank, &go, |_, _| {});
        assert_eq!(result.unwrap_err(), ImportFailure::Empty, "PDF sem texto é recusado com mensagem própria");
        std::fs::remove_file(&blank).unwrap();
    }

    #[test]
    fn ocr_de_pdf_digitalizado() {
        if ocr_languages().is_none() {
            println!("SKIP: tesseract não instalado; verificação de OCR ignorada");
            return;
        }
        let image = cairo::ImageSurface::create(cairo::Format::Rgb24, 1200, 800).unwrap();
        {
            let context = cairo::Context::new(&image).unwrap();
            context.set_source_rgb(1.0, 1.0, 1.0);
            context.paint().unwrap();
            context.set_source_rgb(0.0, 0.0, 0.0);
            context.select_font_face("sans", cairo::FontSlant::Normal, cairo::FontWeight::Normal);
            context.set_font_size(48.0);
            context.move_to(70.0, 200.0);
            context.show_text("Revenue in 2025: 1250 dollars.").unwrap();
        }
        let path = temporary("digitalizado.pdf");
        let surface = cairo::PdfSurface::new(1200.0, 800.0, &path).unwrap();
        let context = cairo::Context::new(&surface).unwrap();
        context.set_source_surface(&image, 0.0, 0.0).unwrap();
        context.paint().unwrap();
        context.show_page().unwrap();
        drop(context);
        surface.finish();

        let scanned = extract(&path, &AtomicBool::new(false), |_, _| {}).expect("OCR");
        assert!(scanned.ocr_pages == 1 && scanned.paragraphs.iter().any(|p| p.original.contains("1250")), "OCR real reconhece texto e números");
        assert!(scanned.paragraphs.iter().all(|p| p.bounds.is_some_and(|b| b.max_x() <= 1200.0 && b.max_y() <= 800.0 && b.width > 50.0)), "área do OCR em coordenadas da página");
        let area = scanned.paragraphs[0].bounds.unwrap();
        assert!(area.y > 540.0 && area.max_y() < 680.0, "área do OCR com y crescendo para cima");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn linhas_do_ocr() {
        let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
                   4\t1\t1\t1\t1\t0\t100\t200\t400\t40\t-1\t\n\
                   5\t1\t1\t1\t1\t1\t100\t200\t180\t40\t96\tReceita\n\
                   5\t1\t1\t1\t1\t2\t300\t204\t200\t36\t91\t2025\n\
                   5\t1\t1\t1\t2\t1\t100\t260\t120\t40\t93\ttotal\n";
        let lines = ocr_lines(tsv, 2.0, 400.0);
        assert!(lines.len() == 2 && lines[0].text == "Receita 2025" && lines[1].text == "total", "palavras do OCR viram linhas");
        assert!(lines[0].x == 50.0 && lines[0].width == 200.0 && lines[0].height == 20.0 && lines[0].y == 400.0 - 120.0, "caixa da linha na escala da página");
    }

    #[test]
    fn pdf_exportado() {
        let mut document = DocumentState::new(vec![Paragraph::new(1, "Hello."), Paragraph::new(2, "Pending.")]);
        document.apply("Olá.", 0);
        let path = temporary("exportado.pdf");
        export(&path, "doc.pdf", "Português", &document).expect("exportação");
        let (pages, text) = text_of(&path);
        assert!(pages == 1 && text.contains("Olá.") && text.contains("Pending.") && !text.contains("Hello."), "PDF exportado contém a tradução e mantém o original pendente");

        let filler = "texto ".repeat(60);
        let mut long = DocumentState::new((0..400).map(|n| Paragraph::new(n / 10 + 1, format!("Parágrafo {n} {filler}"))).collect());
        for index in 0..400 {
            long.apply(&format!("Tradução {index} {filler}"), index);
        }
        export(&path, "longo.pdf", "Português", &long).expect("exportação longa");
        let (pages, text) = text_of(&path);
        assert!(pages > 20 && text.contains("Tradução 399"), "PDF longo pagina até o último parágrafo");
        std::fs::remove_file(&path).unwrap();
    }
}
