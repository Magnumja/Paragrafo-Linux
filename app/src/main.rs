//! Parágrafo para Linux: janela, barra de ferramentas, leitura bilíngue e a ligação com o assistente.

mod chat;
mod pdf;

use adw::prelude::*;
use gtk::{gdk, gio, glib, pango};
use paragrafo_core::assistant;
use paragrafo_core::document::DocumentState;
use paragrafo_core::exporter::{self, ExportFormat};
use paragrafo_core::extraction::{self, union_all, Extracted, ImportFailure, Rect};
use paragrafo_core::language::{self, language_name, LANGUAGES};
use paragrafo_core::library::{Library, RecentDocument, SavedDocument};
use paragrafo_core::ollama::{OllamaClient, OllamaError};
use paragrafo_core::settings::{Settings, DEFAULT_FONT_SIZE, MAX_FONT_SIZE, MIN_FONT_SIZE};
use paragrafo_core::text::{now, prefix};
use paragrafo_core::translate;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub const APP_ID: &str = "io.github.magnumja.Paragrafo";
const LOGO: &[u8] = include_bytes!("../../data/paragrafo-logo.png");
const LOGO_WHITE: &[u8] = include_bytes!("../../data/paragrafo-logo-white.png");
const STYLE: &str = include_str!("style.css");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Importing,
    Translating,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Parallel,
    Translation,
    Original,
}

impl Mode {
    const ALL: [Mode; 3] = [Mode::Parallel, Mode::Translation, Mode::Original];

    fn title(self) -> &'static str {
        match self {
            Mode::Parallel => "Lado a lado",
            Mode::Translation => "Tradução",
            Mode::Original => "PDF original",
        }
    }

    fn key(self) -> &'static str {
        match self {
            Mode::Parallel => "parallel",
            Mode::Translation => "translation",
            Mode::Original => "original",
        }
    }

    fn from_key(key: &str) -> Mode {
        Mode::ALL.into_iter().find(|mode| mode.key() == key).unwrap_or(Mode::Parallel)
    }
}

/// What worker threads report back to the interface.
pub enum Event {
    ImportProgress { run: u64, page: usize, total: usize },
    Imported { run: u64, path: PathBuf, result: Result<Extracted, ImportFailure> },
    TranslationModel { run: u64, model: String },
    Translating { run: u64, index: usize },
    Translated { run: u64, index: usize, text: String },
    TranslationEnded { run: u64, error: Option<String> },
    Chat(chat::ChatEvent),
}

pub struct State {
    pub doc: DocumentState,
    pub filename: String,
    pub path: Option<PathBuf>,
    pub fingerprint: String,
    content_key: String,
    pub page_count: usize,
    pub source: String,
    pub target: String,
    pub detected: Option<String>,
    pub selected_page: usize,
    page_forward: bool,
    pub mode: Mode,
    pub phase: Phase,
    stopping: bool,
    status: String,
    notice: String,
    import_progress: f64,
    /// Paragraph being translated right now.
    active: Option<usize>,
    /// Passage the reader was sent to, e.g. from a source cited in an answer.
    highlighted: Option<Range<usize>>,
    pub pdf: Option<poppler::Document>,
    recents: Vec<RecentDocument>,
    run: u64,
    cancel: Arc<AtomicBool>,
    translated_in_run: usize,
    translation_model: String,
    pub settings: Settings,
}

struct Ui {
    root: gtk::Stack,
    logo: gtk::Image,
    drop_zone: gtk::Box,
    choose: gtk::Button,
    start_progress_box: gtk::Box,
    start_progress: gtk::ProgressBar,
    start_status: gtk::Label,
    start_cancel: gtk::Button,
    recents_box: gtk::Box,
    recents_list: gtk::ListBox,
    home: gtk::Button,
    sidebar_toggle: gtk::ToggleButton,
    modes: [gtk::ToggleButton; 3],
    previous: gtk::Button,
    next: gtk::Button,
    smaller: gtk::Button,
    larger: gtk::Button,
    export: gtk::MenuButton,
    export_buttons: Vec<(ExportFormat, gtk::Button)>,
    open: gtk::Button,
    assistant_toggle: gtk::ToggleButton,
    split: adw::OverlaySplitView,
    inner: adw::OverlaySplitView,
    reading: gtk::Stack,
    file_label: gtk::Label,
    counts_label: gtk::Label,
    source: gtk::DropDown,
    target: gtk::DropDown,
    progress: gtk::ProgressBar,
    progress_label: gtk::Label,
    percent_label: gtk::Label,
    translate: gtk::Button,
    done_label: gtk::Label,
    pages: gtk::ListBox,
    import_progress: gtk::ProgressBar,
    import_cancel: gtk::Button,
    spinner: gtk::Spinner,
    status: gtk::Label,
}

struct RowWidgets {
    container: gtk::Box,
    slot: gtk::Box,
}

#[derive(Default)]
struct PageWidgets {
    rows: HashMap<usize, RowWidgets>,
    scroll: Option<gtk::ScrolledWindow>,
    translate_here: Option<gtk::Button>,
    /// Sidebar rows by page: snippet and progress labels.
    sidebar: HashMap<usize, (gtk::Label, gtk::Label)>,
}

pub struct App {
    pub window: adw::ApplicationWindow,
    ui: Ui,
    pub chat_ui: chat::ChatUi,
    pub state: RefCell<State>,
    pub chat: RefCell<chat::ChatState>,
    pub tx: async_channel::Sender<Event>,
    pub library: Library,
    pub client: OllamaClient,
    /// Set while the code itself changes selections and toggles, so their handlers stay quiet.
    updating: Cell<bool>,
    colors: gtk::CssProvider,
    font: gtk::CssProvider,
    page: RefCell<PageWidgets>,
    pending_open: RefCell<Option<PathBuf>>,
}

// Small widget helpers

pub fn clear(container: &gtk::Box) {
    while let Some(child) = container.first_child() {
        container.remove(&child);
    }
}

pub fn text_label(text: &str, classes: &[&str]) -> gtk::Label {
    let label = gtk::Label::builder().label(text).xalign(0.0).wrap(true).wrap_mode(pango::WrapMode::WordChar).build();
    for class in classes {
        label.add_css_class(class);
    }
    label
}

pub fn section_label(text: &str) -> gtk::Label {
    text_label(&text.to_uppercase(), &["section-label", "dim-label"])
}

pub fn icon_button(icon: &str, title: &str) -> gtk::Button {
    let button = gtk::Button::builder().icon_name(icon).tooltip_text(title).build();
    button.update_property(&[gtk::accessible::Property::Label(title)]);
    button
}

fn labelled_button(label: &str, title: &str) -> gtk::Button {
    let button = gtk::Button::builder().label(label).tooltip_text(title).build();
    button.update_property(&[gtk::accessible::Property::Label(title)]);
    button
}

fn relative_time(opened: f64) -> String {
    let seconds = (now() - opened).max(0.0) as u64;
    match seconds {
        0..=59 => "agora".into(),
        60..=3599 => format!("há {} min", seconds / 60),
        3600..=86_399 => format!("há {} h", seconds / 3600),
        86_400..=172_799 => "ontem".into(),
        _ => format!("há {} dias", seconds / 86_400),
    }
}

fn build_ui() -> Ui {
    // Start screen
    let logo = gtk::Image::builder().pixel_size(140).build();
    logo.update_property(&[gtk::accessible::Property::Label("Parágrafo")]);
    let title = gtk::Label::builder().label("Leia, traduza e pergunte.\nTudo no mesmo documento.").justify(gtk::Justification::Center).wrap(true).build();
    title.add_css_class("start-title");
    let subtitle = gtk::Label::builder()
        .label("Abra um PDF para ler o original ao lado da tradução\ne fazer perguntas que apontam para o trecho exato.")
        .justify(gtk::Justification::Center)
        .wrap(true)
        .margin_top(14)
        .build();
    subtitle.add_css_class("dim-label");

    let start_progress = gtk::ProgressBar::builder().width_request(260).halign(gtk::Align::Center).build();
    let start_status = gtk::Label::new(None);
    start_status.add_css_class("dim-label");
    let start_cancel = gtk::Button::builder().label("Cancelar").halign(gtk::Align::Center).build();
    let start_progress_box = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(10).margin_top(30).visible(false).build();
    start_progress_box.append(&start_progress);
    start_progress_box.append(&start_status);
    start_progress_box.append(&start_cancel);

    let choose = gtk::Button::builder().label("Escolher PDF").halign(gtk::Align::Center).build();
    choose.add_css_class("suggested-action");
    choose.add_css_class("pill");
    let drop_icon = gtk::Image::builder().icon_name("document-open-symbolic").pixel_size(28).build();
    drop_icon.add_css_class("accent");
    let drop_hint = gtk::Label::new(Some("ou arraste um PDF para cá"));
    drop_hint.add_css_class("dim-label");
    drop_hint.add_css_class("caption");
    let drop_zone = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(12).margin_top(28).build();
    drop_zone.add_css_class("drop-zone");
    drop_zone.append(&drop_icon);
    drop_zone.append(&choose);
    drop_zone.append(&drop_hint);

    let recents_list = gtk::ListBox::builder().selection_mode(gtk::SelectionMode::None).build();
    recents_list.add_css_class("boxed-list");
    let recents_box = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(6).margin_top(36).visible(false).build();
    recents_box.append(&section_label("Continuar de onde parou"));
    recents_box.append(&recents_list);

    let features = gtk::Label::builder()
        .label("Original e tradução lado a lado  ·  Perguntas com Llama local  ·  Nada sai do seu computador")
        .justify(gtk::Justification::Center)
        .wrap(true)
        .margin_top(36)
        .build();
    features.add_css_class("dim-label");
    features.add_css_class("caption");

    let start_column = gtk::Box::builder().orientation(gtk::Orientation::Vertical).valign(gtk::Align::Center).margin_top(40).margin_bottom(40).margin_start(24).margin_end(24).build();
    for widget in [logo.upcast_ref::<gtk::Widget>(), title.upcast_ref(), subtitle.upcast_ref(), start_progress_box.upcast_ref(), drop_zone.upcast_ref(), recents_box.upcast_ref(), features.upcast_ref()] {
        start_column.append(widget);
    }
    let start_clamp = adw::Clamp::builder().maximum_size(520).child(&start_column).build();
    let start_scroll = gtk::ScrolledWindow::builder().hscrollbar_policy(gtk::PolicyType::Never).child(&start_clamp).build();
    let start_view = adw::ToolbarView::new();
    start_view.add_top_bar(&adw::HeaderBar::new());
    start_view.set_content(Some(&start_scroll));

    // Toolbar
    let home = icon_button("user-home-symbolic", "Voltar à tela inicial. O progresso fica salvo.");
    let sidebar_toggle = gtk::ToggleButton::builder()
        .label("Páginas").tooltip_text("Mostrar ou ocultar as páginas").active(true).build();
    sidebar_toggle.update_property(&[gtk::accessible::Property::Label("Páginas")]);
    let modes = Mode::ALL.map(|mode| gtk::ToggleButton::builder().label(mode.title()).build());
    modes[1].set_group(Some(&modes[0]));
    modes[2].set_group(Some(&modes[0]));
    let mode_box = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    mode_box.add_css_class("linked");
    mode_box.update_property(&[gtk::accessible::Property::Label("Modo de leitura")]);
    for button in &modes {
        mode_box.append(button);
    }
    let previous = icon_button("go-previous-symbolic", "Página anterior (← ou Ctrl+[)");
    let next = icon_button("go-next-symbolic", "Próxima página (→ ou Ctrl+])");
    let pager = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    pager.add_css_class("linked");
    pager.append(&previous);
    pager.append(&next);
    let smaller = labelled_button("A−", "Diminuir texto (Ctrl+−)");
    let larger = labelled_button("A+", "Aumentar texto (Ctrl++)");
    let sizes = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    sizes.add_css_class("linked");
    sizes.append(&smaller);
    sizes.append(&larger);
    let export_list = gtk::Box::new(gtk::Orientation::Vertical, 2);
    let export_buttons: Vec<(ExportFormat, gtk::Button)> = ExportFormat::ALL
        .into_iter()
        .map(|format| {
            let button = gtk::Button::builder().label(format.title()).build();
            button.add_css_class("flat");
            export_list.append(&button);
            (format, button)
        })
        .collect();
    let export_popover = gtk::Popover::builder().child(&export_list).build();
    let export = gtk::MenuButton::builder().icon_name("document-save-as-symbolic").tooltip_text("Exportar a tradução").popover(&export_popover).build();
    export.update_property(&[gtk::accessible::Property::Label("Exportar")]);
    let open = icon_button("document-open-symbolic", "Abrir PDF (Ctrl+O)");
    let assistant_toggle = gtk::ToggleButton::builder().label("Perguntar").tooltip_text("Mostrar ou ocultar o assistente (Ctrl+K)").build();

    let header = adw::HeaderBar::new();
    header.pack_start(&home);
    header.pack_start(&sidebar_toggle);
    header.set_title_widget(Some(&mode_box));
    header.pack_end(&assistant_toggle);
    header.pack_end(&open);
    header.pack_end(&export);
    header.pack_end(&sizes);
    header.pack_end(&pager);

    // Sidebar
    let file_label = gtk::Label::builder().xalign(0.0).wrap(true).wrap_mode(pango::WrapMode::WordChar).lines(2).ellipsize(pango::EllipsizeMode::Middle).build();
    file_label.add_css_class("heading");
    let counts_label = text_label("", &["caption", "dim-label"]);
    let source = gtk::DropDown::from_strings(&[]);
    source.set_hexpand(true);
    source.update_property(&[gtk::accessible::Property::Label("Idioma de origem")]);
    let target = gtk::DropDown::from_strings(&[]);
    target.set_hexpand(true);
    target.update_property(&[gtk::accessible::Property::Label("Idioma de destino")]);
    let languages = gtk::Grid::builder().row_spacing(6).column_spacing(8).build();
    languages.attach(&text_label("De", &["caption", "dim-label"]), 0, 0, 1, 1);
    languages.attach(&source, 1, 0, 1, 1);
    languages.attach(&text_label("Para", &["caption", "dim-label"]), 0, 1, 1, 1);
    languages.attach(&target, 1, 1, 1, 1);
    let progress = gtk::ProgressBar::new();
    progress.update_property(&[gtk::accessible::Property::Label("Progresso da tradução")]);
    let progress_label = text_label("", &["caption", "dim-label"]);
    progress_label.set_hexpand(true);
    let percent_label = text_label("", &["caption", "dim-label", "numeric"]);
    let progress_row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    progress_row.append(&progress_label);
    progress_row.append(&percent_label);
    let translate = gtk::Button::builder().label("Traduzir documento").tooltip_text("Começa pela página que você está lendo").build();
    let done_label = gtk::Label::builder().label("Tradução concluída").visible(false).build();
    done_label.add_css_class("done-label");
    let controls = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(10).margin_top(8).margin_bottom(12).margin_start(14).margin_end(14).build();
    for widget in [file_label.upcast_ref::<gtk::Widget>(), counts_label.upcast_ref(), languages.upcast_ref(), progress.upcast_ref(), progress_row.upcast_ref(), translate.upcast_ref(), done_label.upcast_ref()] {
        controls.append(widget);
    }
    let pages = gtk::ListBox::builder().selection_mode(gtk::SelectionMode::Single).build();
    pages.add_css_class("navigation-sidebar");
    pages.update_property(&[gtk::accessible::Property::Label("Páginas")]);
    let pages_scroll = gtk::ScrolledWindow::builder().hscrollbar_policy(gtk::PolicyType::Never).vexpand(true).child(&pages).build();
    let import_progress = gtk::ProgressBar::builder().visible(false).build();
    let import_cancel = gtk::Button::builder().label("Cancelar importação").visible(false).halign(gtk::Align::Start).build();
    let spinner = gtk::Spinner::new();
    let status = text_label("", &["caption", "dim-label"]);
    status.set_hexpand(true);
    let status_row = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    status_row.append(&spinner);
    status_row.append(&status);
    let privacy = text_label("Tudo é processado neste computador", &["caption", "dim-label"]);
    let footer = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(6).margin_top(10).margin_bottom(10).margin_start(14).margin_end(14).build();
    footer.append(&import_progress);
    footer.append(&import_cancel);
    footer.append(&status_row);
    footer.append(&privacy);
    let sidebar = gtk::Box::new(gtk::Orientation::Vertical, 0);
    sidebar.append(&controls);
    sidebar.append(&gtk::Separator::new(gtk::Orientation::Horizontal));
    sidebar.append(&pages_scroll);
    sidebar.append(&gtk::Separator::new(gtk::Orientation::Horizontal));
    sidebar.append(&footer);

    // Reading area and assistant
    let reading = gtk::Stack::builder().transition_duration(260).hexpand(true).vexpand(true).build();
    let inner = adw::OverlaySplitView::new();
    inner.set_sidebar_position(gtk::PackType::End);
    inner.set_min_sidebar_width(300.0);
    inner.set_max_sidebar_width(540.0);
    inner.set_sidebar_width_fraction(0.32);
    inner.set_content(Some(&reading));
    let split = adw::OverlaySplitView::new();
    split.set_min_sidebar_width(230.0);
    split.set_max_sidebar_width(360.0);
    split.set_sidebar_width_fraction(0.2);
    split.set_sidebar(Some(&sidebar));
    split.set_content(Some(&inner));
    let reader_view = adw::ToolbarView::new();
    reader_view.add_top_bar(&header);
    reader_view.set_content(Some(&split));

    let root = gtk::Stack::builder().transition_type(gtk::StackTransitionType::Crossfade).build();
    root.add_named(&start_view, Some("start"));
    root.add_named(&reader_view, Some("reader"));

    Ui {
        root,
        logo,
        drop_zone,
        choose,
        start_progress_box,
        start_progress,
        start_status,
        start_cancel,
        recents_box,
        recents_list,
        home,
        sidebar_toggle,
        modes,
        previous,
        next,
        smaller,
        larger,
        export,
        export_buttons,
        open,
        assistant_toggle,
        split,
        inner,
        reading,
        file_label,
        counts_label,
        source,
        target,
        progress,
        progress_label,
        percent_label,
        translate,
        done_label,
        pages,
        import_progress,
        import_cancel,
        spinner,
        status,
    }
}

impl App {
    fn new(application: &adw::Application) -> Rc<Self> {
        let settings = Settings::load();
        let library = Library::default();
        let window = adw::ApplicationWindow::builder().application(application).title("Parágrafo").default_width(1400).default_height(880).width_request(720).height_request(520).build();
        let ui = build_ui();
        let (chat_ui, panel) = chat::build();
        ui.inner.set_sidebar(Some(&panel));
        window.set_content(Some(&ui.root));
        let (tx, rx) = async_channel::unbounded::<Event>();
        let state = State {
            doc: DocumentState::default(),
            filename: String::new(),
            path: None,
            fingerprint: String::new(),
            content_key: String::new(),
            page_count: 0,
            source: "auto".into(),
            target: "pt".into(),
            detected: None,
            selected_page: 1,
            page_forward: true,
            mode: Mode::from_key(&settings.reading_mode),
            phase: Phase::Idle,
            stopping: false,
            status: "Pronto para um novo documento.".into(),
            notice: String::new(),
            import_progress: 0.0,
            active: None,
            highlighted: None,
            pdf: None,
            recents: library.recents(),
            run: 0,
            cancel: Arc::new(AtomicBool::new(false)),
            translated_in_run: 0,
            translation_model: String::new(),
            settings: settings.clone(),
        };
        let app = Rc::new(App {
            window,
            ui,
            chat_ui,
            state: RefCell::new(state),
            chat: RefCell::new(chat::ChatState::new(&settings.chat_model)),
            tx,
            library,
            client: OllamaClient::default(),
            updating: Cell::new(false),
            colors: gtk::CssProvider::new(),
            font: gtk::CssProvider::new(),
            page: RefCell::new(PageWidgets::default()),
            pending_open: RefCell::new(None),
        });
        app.install_styles();
        app.connect(application);
        app.chat_connect();
        let receiver = app.clone();
        glib::spawn_future_local(async move {
            while let Ok(event) = rx.recv().await {
                receiver.handle(event);
            }
        });
        app.refresh_languages();
        app.refresh();
        app.chat_refresh();
        app
    }

    // Styles

    fn install_styles(self: &Rc<Self>) {
        let Some(display) = gdk::Display::default() else { return };
        let base = gtk::CssProvider::new();
        base.load_from_string(STYLE);
        for provider in [&base, &self.colors, &self.font] {
            gtk::style_context_add_provider_for_display(&display, provider, gtk::STYLE_PROVIDER_PRIORITY_APPLICATION);
        }
        let manager = adw::StyleManager::default();
        let app = self.clone();
        manager.connect_dark_notify(move |_| app.apply_colors());
        self.apply_colors();
        self.apply_font();
    }

    /// The moss green of the Mac app, lighter on dark backgrounds.
    fn apply_colors(&self) {
        let dark = adw::StyleManager::default().is_dark();
        let (accent, background) = if dark { ("#8cc2a6", "#3f7a62") } else { ("#2f5a48", "#386654") };
        self.colors.load_from_string(&format!(
            "@define-color accent_color {accent}; @define-color accent_bg_color {background}; @define-color accent_fg_color #ffffff;"
        ));
        if let Ok(texture) = gdk::Texture::from_bytes(&glib::Bytes::from_static(if dark { LOGO_WHITE } else { LOGO })) {
            self.ui.logo.set_paintable(Some(&texture));
        }
    }

    fn apply_font(&self) {
        let size = self.state.borrow().settings.font_size;
        self.font.load_from_string(&format!(".reading-text {{ font-family: serif; font-size: {size}px; line-height: 1.5; }}"));
    }

    // Wiring

    fn action(self: &Rc<Self>, application: &adw::Application, name: &str, accelerators: &[&str], run: impl Fn(&Rc<App>) + 'static) {
        let action = gio::SimpleAction::new(name, None);
        let app = self.clone();
        action.connect_activate(move |_, _| run(&app));
        self.window.add_action(&action);
        application.set_accels_for_action(&format!("win.{name}"), accelerators);
    }

    fn connect(self: &Rc<Self>, application: &adw::Application) {
        let ui = &self.ui;
        self.action(application, "open", &["<Control>o"], |app| app.choose_file());
        self.action(application, "assistant", &["<Control>k"], |app| app.toggle_assistant());
        self.action(application, "previous", &["<Control>bracketleft"], |app| app.move_page(-1));
        self.action(application, "next", &["<Control>bracketright"], |app| app.move_page(1));
        self.action(application, "larger", &["<Control>plus", "<Control>equal", "<Control>KP_Add"], |app| app.change_font(1.0));
        self.action(application, "smaller", &["<Control>minus", "<Control>KP_Subtract"], |app| app.change_font(-1.0));
        self.action(application, "reset-size", &["<Control>0"], |app| app.change_font(0.0));

        let click = |button: &gtk::Button, run: fn(&Rc<App>)| {
            let app = self.clone();
            button.connect_clicked(move |_| run(&app));
        };
        click(&ui.choose, |app| app.choose_file());
        click(&ui.open, |app| app.choose_file());
        click(&ui.start_cancel, |app| app.stop());
        click(&ui.import_cancel, |app| app.stop());
        click(&ui.home, |app| app.close());
        click(&ui.previous, |app| app.move_page(-1));
        click(&ui.next, |app| app.move_page(1));
        click(&ui.smaller, |app| app.change_font(-1.0));
        click(&ui.larger, |app| app.change_font(1.0));
        click(&ui.translate, |app| {
            if app.state.borrow().phase == Phase::Translating {
                app.stop();
            } else {
                app.start_translation();
            }
        });
        for (format, button) in &ui.export_buttons {
            let (app, format) = (self.clone(), *format);
            button.connect_clicked(move |_| {
                app.ui.export.popdown();
                app.export(format);
            });
        }
        for (mode, button) in Mode::ALL.into_iter().zip(&ui.modes) {
            let app = self.clone();
            button.connect_toggled(move |button| {
                if button.is_active() && !app.updating.get() {
                    app.set_mode(mode);
                }
            });
        }
        let app = self.clone();
        ui.sidebar_toggle.connect_toggled(move |button| app.ui.split.set_show_sidebar(button.is_active()));
        let app = self.clone();
        ui.assistant_toggle.connect_toggled(move |button| {
            if !app.updating.get() {
                app.show_assistant(button.is_active());
            }
        });
        let app = self.clone();
        ui.pages.connect_row_selected(move |_, row| {
            if app.updating.get() {
                return;
            }
            let page = row.and_then(|row| app.state.borrow().doc.pages().get(row.index().max(0) as usize).copied());
            if let Some(page) = page {
                app.select_page(page);
            }
        });
        let app = self.clone();
        ui.source.connect_selected_notify(move |dropdown| {
            if app.updating.get() {
                return;
            }
            let code = match dropdown.selected() {
                0 => "auto".to_string(),
                position => LANGUAGES.get(position as usize - 1).map_or("auto", |language| language.code).to_string(),
            };
            app.choose_language(Some(code), None);
        });
        let app = self.clone();
        ui.target.connect_selected_notify(move |dropdown| {
            if app.updating.get() {
                return;
            }
            if let Some(language) = LANGUAGES.get(dropdown.selected() as usize) {
                app.choose_language(None, Some(language.code.to_string()));
            }
        });
        let app = self.clone();
        ui.recents_list.connect_row_activated(move |_, row| {
            let recent = app.state.borrow().recents.get(row.index().max(0) as usize).cloned();
            if let Some(recent) = recent {
                app.open_recent(&recent);
            }
        });

        // A PDF dropped anywhere on the window opens it.
        let drop = gtk::DropTarget::new(gdk::FileList::static_type(), gdk::DragAction::COPY);
        let app = self.clone();
        drop.connect_enter(move |_, _, _| {
            app.ui.drop_zone.add_css_class("targeted");
            gdk::DragAction::COPY
        });
        let app = self.clone();
        drop.connect_leave(move |_| app.ui.drop_zone.remove_css_class("targeted"));
        let app = self.clone();
        drop.connect_drop(move |_, value, _, _| {
            app.ui.drop_zone.remove_css_class("targeted");
            let path = value.get::<gdk::FileList>().ok().and_then(|list| list.files().into_iter().filter_map(|file| file.path()).find(|path| is_pdf(path)));
            match path {
                Some(path) if app.state.borrow().phase == Phase::Idle => {
                    app.load(path);
                    true
                }
                _ => false,
            }
        });
        self.window.add_controller(drop);

        // ← → and Page Up / Page Down turn the page, Home / End jump to the first / last one.
        let keys = gtk::EventControllerKey::new();
        keys.set_propagation_phase(gtk::PropagationPhase::Capture);
        let app = self.clone();
        keys.connect_key_pressed(move |_, key, _, modifiers| app.key(key, modifiers));
        self.window.add_controller(keys);

        // A horizontal two-finger swipe over the text turns the page.
        let swipe = gtk::EventControllerScroll::new(gtk::EventControllerScrollFlags::BOTH_AXES);
        swipe.set_propagation_phase(gtk::PropagationPhase::Capture);
        let travel = Rc::new(Cell::new(0.0f64));
        let turned = Rc::new(Cell::new(false));
        let (begin_travel, begin_turned) = (travel.clone(), turned.clone());
        swipe.connect_scroll_begin(move |_| {
            begin_travel.set(0.0);
            begin_turned.set(false);
        });
        let app = self.clone();
        swipe.connect_scroll(move |controller, dx, dy| {
            if controller.unit() != gdk::ScrollUnit::Surface || dx.abs() <= dy.abs() * 1.5 {
                return glib::Propagation::Proceed;
            }
            travel.set(travel.get() + dx);
            if !turned.get() && travel.get().abs() > 110.0 {
                turned.set(true);
                app.move_page(if travel.get() > 0.0 { 1 } else { -1 });
            }
            glib::Propagation::Stop
        });
        ui.reading.add_controller(swipe);

        let app = self.clone();
        self.window.connect_close_request(move |_| {
            app.state.borrow().cancel.store(true, Ordering::Relaxed);
            app.chat_cancel_all();
            app.persist();
            glib::Propagation::Proceed
        });
    }

    fn key(self: &Rc<Self>, key: gdk::Key, modifiers: gdk::ModifierType) -> glib::Propagation {
        let held = modifiers & (gdk::ModifierType::CONTROL_MASK | gdk::ModifierType::SHIFT_MASK | gdk::ModifierType::ALT_MASK | gdk::ModifierType::SUPER_MASK);
        if !held.is_empty() || self.state.borrow().doc.is_empty() {
            return glib::Propagation::Proceed;
        }
        // Never while typing, in a menu or in the PDF peek: those handle their own keys.
        if let Some(focus) = gtk::prelude::GtkWindowExt::focus(&self.window) {
            let typing = focus.is::<gtk::Text>() || focus.is::<gtk::TextView>() || focus.is::<gtk::Entry>();
            if typing || focus.ancestor(gtk::Popover::static_type()).is_some() || focus.ancestor(gtk::DropDown::static_type()).is_some() {
                return glib::Propagation::Proceed;
            }
        }
        let (first, last) = {
            let state = self.state.borrow();
            (state.doc.pages().first().copied(), state.doc.pages().last().copied())
        };
        match key {
            gdk::Key::Left | gdk::Key::Page_Up => self.move_page(-1),
            gdk::Key::Right | gdk::Key::Page_Down => self.move_page(1),
            gdk::Key::Home => self.select_page(first.unwrap_or(1)),
            gdk::Key::End => self.select_page(last.unwrap_or(1)),
            _ => return glib::Propagation::Proceed,
        }
        glib::Propagation::Stop
    }

    // Events from worker threads

    fn handle(self: &Rc<Self>, event: Event) {
        match event {
            Event::ImportProgress { run, page, total } => {
                {
                    let mut state = self.state.borrow_mut();
                    if state.run != run || state.phase != Phase::Importing || state.stopping {
                        return;
                    }
                    state.import_progress = page as f64 / total.max(1) as f64;
                    state.status = format!("Lendo página {page} de {total}…");
                }
                self.refresh();
            }
            Event::Imported { run, path, result } => self.imported(run, path, result),
            Event::TranslationModel { run, model } => {
                let mut state = self.state.borrow_mut();
                if state.run == run {
                    state.translation_model = model;
                }
            }
            Event::Translating { run, index } => {
                let previous = {
                    let mut state = self.state.borrow_mut();
                    if state.run != run || state.phase != Phase::Translating {
                        return;
                    }
                    state.status = format!("Traduzindo com {} · {} de {}", state.translation_model, state.doc.completed() + 1, state.doc.count());
                    state.active.replace(index)
                };
                if let Some(previous) = previous {
                    self.update_row(previous);
                }
                self.update_row(index);
                self.refresh_sidebar();
            }
            Event::Translated { run, index, text } => {
                let save = {
                    let mut state = self.state.borrow_mut();
                    if state.run != run {
                        return;
                    }
                    state.doc.apply(&text, index);
                    state.translated_in_run += 1;
                    state.translated_in_run.is_multiple_of(20)
                };
                if save {
                    self.persist();
                }
                self.update_row(index);
                self.refresh_sidebar();
                self.refresh_toolbar();
            }
            Event::TranslationEnded { run, error } => self.translation_ended(run, error),
            Event::Chat(event) => self.chat_handle(event),
        }
    }

    // Opening documents

    fn choose_file(self: &Rc<Self>) {
        if self.state.borrow().phase != Phase::Idle {
            return;
        }
        let filter = gtk::FileFilter::new();
        filter.set_name(Some("Documentos PDF"));
        filter.add_mime_type("application/pdf");
        filter.add_suffix("pdf");
        let filters = gio::ListStore::new::<gtk::FileFilter>();
        filters.append(&filter);
        let dialog = gtk::FileDialog::builder().title("Abrir PDF").modal(true).filters(&filters).build();
        let app = self.clone();
        dialog.open(Some(&self.window), gio::Cancellable::NONE, move |result| {
            if let Some(path) = result.ok().and_then(|file| file.path()) {
                app.load(path);
            }
        });
    }

    /// A PDF requested while another import or a translation is running waits for it to finish.
    pub fn load(self: &Rc<Self>, path: PathBuf) {
        let (run, cancel) = {
            let mut state = self.state.borrow_mut();
            if state.phase != Phase::Idle {
                self.pending_open.replace(Some(path));
                return;
            }
            state.run += 1;
            state.phase = Phase::Importing;
            state.stopping = false;
            state.import_progress = 0.0;
            state.status = format!("Abrindo {}…", file_name(&path));
            state.cancel = Arc::new(AtomicBool::new(false));
            (state.run, state.cancel.clone())
        };
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let progress = tx.clone();
            let result = pdf::extract(&path, &cancel, |page, total| {
                let _ = progress.send_blocking(Event::ImportProgress { run, page, total });
            });
            let _ = tx.send_blocking(Event::Imported { run, path, result });
        });
        self.refresh();
    }

    fn imported(self: &Rc<Self>, run: u64, path: PathBuf, result: Result<Extracted, ImportFailure>) {
        if self.state.borrow().run != run {
            return;
        }
        let failure = match result {
            Ok(extracted) => {
                self.adopt(path, extracted);
                None
            }
            Err(failure) => Some(failure),
        };
        {
            let mut state = self.state.borrow_mut();
            state.phase = Phase::Idle;
            state.stopping = false;
            match &failure {
                Some(ImportFailure::Cancelled) => state.status = "Importação cancelada.".into(),
                Some(_) => state.status = "Não foi possível importar o PDF.".into(),
                None => {}
            }
        }
        self.refresh();
        match failure {
            Some(ImportFailure::Cancelled) | None => {}
            Some(failure) => self.alert(&failure.to_string()),
        }
        self.open_pending();
    }

    fn adopt(self: &Rc<Self>, path: PathBuf, result: Extracted) {
        let mut paragraphs = result.paragraphs;
        let detected = language::detect_language(&paragraphs);
        let key = extraction::content_key(&paragraphs);
        // Reopening the same file brings back the work already done.
        let saved = self.library.load(&result.fingerprint);
        let mut restored = 0;
        let mut languages = None;
        match saved {
            Some(saved) if saved.content_key.as_deref() == Some(key.as_str()) && saved.translations.len() == paragraphs.len() => {
                for (paragraph, translation) in paragraphs.iter_mut().zip(&saved.translations) {
                    paragraph.translated = translation.clone();
                }
                restored = saved.translations.iter().filter(|text| !text.is_empty()).count();
                languages = Some((saved.source, saved.target));
            }
            // Same file, different extraction: what was saved no longer lines up with the text.
            Some(_) => self.library.remove(&result.fingerprint),
            None => {}
        }
        {
            let mut state = self.state.borrow_mut();
            state.detected = detected;
            match languages {
                Some((source, target)) => {
                    state.source = source;
                    state.target = target;
                }
                None => state.source = "auto".into(),
            }
            state.content_key = key;
            state.doc = DocumentState::new(paragraphs);
            state.selected_page = state.doc.pages().first().copied().unwrap_or(1);
            state.highlighted = None;
            state.active = None;
            state.filename = file_name(&path);
            state.page_count = result.page_count;
            state.pdf = pdf::open(&path);
            state.path = Some(path);
            state.fingerprint = result.fingerprint;
            state.status = if restored > 0 {
                format!("{restored} de {} parágrafos já traduzidos foram restaurados.", state.doc.count())
            } else {
                format!("{} parágrafos prontos para traduzir.", state.doc.count())
            };
            let mut notice = Vec::new();
            if result.ocr_pages > 0 {
                notice.push(format!("OCR usado em {} página(s). Confira o texto reconhecido.", result.ocr_pages));
            }
            if result.ocr_missing {
                notice.push("Há páginas digitalizadas: instale o tesseract-ocr para reconhecer o texto delas.".to_string());
            }
            if !result.empty_pages.is_empty() {
                notice.push(format!("Sem texto nas páginas: {}.", result.empty_pages.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")));
            }
            state.notice = notice.join(" ");
        }
        self.persist();
        self.refresh_languages();
        self.rebuild_pages();
        self.show_page(false);
        self.ui.root.set_visible_child_name("reader");
        let (show, paragraphs, fingerprint) = {
            let state = self.state.borrow();
            (state.settings.show_assistant, state.doc.paragraphs().to_vec(), state.fingerprint.clone())
        };
        self.chat_attach(paragraphs, &fingerprint);
        self.show_assistant(show);
    }

    fn open_pending(self: &Rc<Self>) {
        if self.state.borrow().phase != Phase::Idle {
            return;
        }
        if let Some(path) = self.pending_open.take() {
            self.load(path);
        }
    }

    fn open_recent(self: &Rc<Self>, recent: &RecentDocument) {
        let path = PathBuf::from(&recent.path);
        if !path.is_file() {
            self.forget(&recent.id);
            let folder = path.parent().map(|parent| parent.display().to_string()).unwrap_or_default();
            self.alert(&format!("“{}” não está mais em {folder}. Abra o arquivo novamente.", recent.name));
            return;
        }
        self.load(path);
    }

    fn forget(self: &Rc<Self>, fingerprint: &str) {
        self.library.remove(fingerprint);
        self.state.borrow_mut().recents = self.library.recents();
        self.refresh_start();
    }

    /// Back to the start screen. The work stays saved.
    fn close(self: &Rc<Self>) {
        if self.state.borrow().phase != Phase::Idle {
            return;
        }
        self.persist();
        {
            let mut state = self.state.borrow_mut();
            state.doc = DocumentState::default();
            state.filename.clear();
            state.fingerprint.clear();
            state.pdf = None;
            state.path = None;
            state.highlighted = None;
            state.notice.clear();
            state.status = "Pronto para um novo documento.".into();
        }
        self.chat_attach(Vec::new(), "");
        *self.page.borrow_mut() = PageWidgets::default();
        while let Some(child) = self.ui.reading.first_child() {
            self.ui.reading.remove(&child);
        }
        self.ui.root.set_visible_child_name("start");
        self.refresh();
    }

    fn stop(self: &Rc<Self>) {
        {
            let mut state = self.state.borrow_mut();
            match state.phase {
                Phase::Idle => return,
                Phase::Importing => state.status = "Cancelando após a página atual…".into(),
                Phase::Translating => state.status = "Pausando…".into(),
            }
            state.stopping = true;
            state.cancel.store(true, Ordering::Relaxed);
        }
        self.refresh();
    }

    // Languages and translation

    fn choose_language(self: &Rc<Self>, source: Option<String>, target: Option<String>) {
        {
            let mut state = self.state.borrow_mut();
            if state.phase != Phase::Idle {
                return;
            }
            let source = source.unwrap_or_else(|| state.source.clone());
            let target = target.unwrap_or_else(|| state.target.clone());
            if source == state.source && target == state.target {
                return;
            }
            state.source = source;
            state.target = target;
            // A translation belongs to its pair of languages.
            state.doc.clear_translations();
            state.status = "Idiomas alterados. Pronto para traduzir.".into();
        }
        self.persist();
        self.rebuild_pages();
        self.show_page(false);
        self.refresh();
    }

    fn start_translation(self: &Rc<Self>) {
        let mut complaint = None;
        let job = {
            let mut state = self.state.borrow_mut();
            if state.phase != Phase::Idle || state.doc.is_empty() || state.doc.completed() >= state.doc.count() {
                return;
            }
            let origin = if state.source == "auto" { state.detected.clone() } else { Some(state.source.clone()) };
            let has_text = state.doc.paragraphs().iter().any(|paragraph| language::needs_translation(&paragraph.original));
            match &origin {
                Some(origin) if language::same_language(origin, &state.target) => {
                    complaint = Some("A origem e o destino são o mesmo idioma. Se o documento está em inglês, escolha Inglês como origem e Português como destino.");
                    state.status = "Escolha idiomas diferentes para traduzir.".into();
                    None
                }
                None if has_text => {
                    complaint = Some("Não foi possível identificar a língua do documento. Escolha o idioma de origem na barra lateral.");
                    None
                }
                _ => {
                    // Preserve numbers and punctuation without asking a model to translate them.
                    let untouched: Vec<(usize, String)> = state
                        .doc
                        .paragraphs()
                        .iter()
                        .enumerate()
                        .filter(|(_, paragraph)| !language::needs_translation(&paragraph.original))
                        .map(|(index, paragraph)| (index, paragraph.original.clone()))
                        .collect();
                    for (index, text) in untouched {
                        state.doc.apply(&text, index);
                    }
                    match origin {
                        Some(origin) if state.doc.completed() < state.doc.count() => {
                            state.phase = Phase::Translating;
                            state.stopping = false;
                            state.run += 1;
                            state.translated_in_run = 0;
                            state.cancel = Arc::new(AtomicBool::new(false));
                            state.status = format!("Preparando {} → {}…", language_name(&origin), language_name(&state.target));
                            let pending: Vec<(usize, String)> =
                                state.doc.pending_order(state.selected_page).into_iter().map(|index| (index, state.doc.paragraphs()[index].original.clone())).collect();
                            Some((state.run, state.cancel.clone(), origin, state.target.clone(), pending))
                        }
                        _ => {
                            state.status = "Documento sem texto para traduzir. Números e símbolos foram preservados.".into();
                            None
                        }
                    }
                }
            }
        };
        if let Some(message) = complaint {
            self.refresh();
            self.alert(message);
            return;
        }
        let Some((run, cancel, origin, target, pending)) = job else {
            self.persist();
            self.show_page(false);
            self.refresh();
            return;
        };
        let (tx, client) = (self.tx.clone(), self.client.clone());
        let chat_model = self.chat.borrow().chat_model.clone();
        std::thread::spawn(move || {
            let send = |event: Event| {
                let _ = tx.send_blocking(event);
            };
            let checked = assistant::check(&client, &chat_model, true);
            let (availability, installed, chat_models, chosen) = (checked.availability.clone(), checked.installed.clone(), checked.chat_models.clone(), checked.chat_model.clone());
            send(Event::Chat(chat::ChatEvent::Checked(checked)));
            if availability == assistant::Availability::ServerOff {
                let message = "O Ollama não está em execução. A tradução é feita por um modelo local: instale o Ollama (ollama.com/download), inicie-o e tente de novo.";
                return send(Event::TranslationEnded { run, error: Some(message.into()) });
            }
            let Some(model) = translate::pick_model(&installed, &chat_models, &chosen) else {
                let message = "Nenhum modelo para traduzir está instalado no Ollama. Baixe um, por exemplo: ollama pull translategemma:4b";
                return send(Event::TranslationEnded { run, error: Some(message.into()) });
            };
            send(Event::TranslationModel { run, model: model.clone() });
            // Sequential calls keep pause boundaries exact; the page being read goes first.
            for (index, text) in pending {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                send(Event::Translating { run, index });
                match translate::translate(&client, &model, &origin, &target, &text, &cancel) {
                    Ok(translated) => send(Event::Translated { run, index, text: translated }),
                    Err(OllamaError::Cancelled) => break,
                    Err(error) => return send(Event::TranslationEnded { run, error: Some(error.to_string()) }),
                }
            }
            send(Event::TranslationEnded { run, error: None });
        });
        self.rebuild_pages();
        self.show_page(false);
        self.refresh();
    }

    fn translation_ended(self: &Rc<Self>, run: u64, error: Option<String>) {
        let previous = {
            let mut state = self.state.borrow_mut();
            if state.run != run || state.phase != Phase::Translating {
                return;
            }
            state.status = match (&error, state.doc.is_complete()) {
                (Some(_), _) => "Não foi possível concluir a tradução.".into(),
                (None, true) => "Tradução concluída. Revise comparando com o original.".into(),
                (None, false) => "Tradução pausada. Seu progresso foi mantido.".into(),
            };
            state.phase = Phase::Idle;
            state.stopping = false;
            state.active.take()
        };
        self.persist();
        if let Some(previous) = previous {
            self.update_row(previous);
        }
        self.refresh();
        if let Some(error) = error {
            self.alert(&format!("Não foi possível traduzir. O progresso foi mantido.\n\n{error}"));
        }
        self.open_pending();
    }

    // Navigation

    fn select_page(self: &Rc<Self>, page: usize) {
        {
            let mut state = self.state.borrow_mut();
            if state.selected_page == page || state.doc.page_range(page).is_none() {
                return;
            }
            state.page_forward = page > state.selected_page;
            state.selected_page = page;
        }
        self.show_page(true);
        self.refresh_toolbar();
        self.sync_page_selection();
    }

    fn move_page(self: &Rc<Self>, offset: i64) {
        let target = {
            let state = self.state.borrow();
            let pages = state.doc.pages();
            pages
                .iter()
                .position(|&page| page == state.selected_page)
                .map(|at| at as i64 + offset)
                .filter(|&at| at >= 0)
                .and_then(|at| pages.get(at as usize).copied())
        };
        if let Some(page) = target {
            self.select_page(page);
        }
    }

    /// Jump to a passage and mark it, e.g. from a source cited in an answer.
    pub fn reveal_paragraphs(self: &Rc<Self>, range: Range<usize>) {
        {
            let mut state = self.state.borrow_mut();
            let range = range.start.min(state.doc.count())..range.end.min(state.doc.count());
            if range.is_empty() {
                return;
            }
            let page = state.doc.paragraphs()[range.start].page;
            state.page_forward = page >= state.selected_page;
            state.selected_page = page;
            state.highlighted = Some(range);
        }
        self.show_page(false);
        self.refresh_toolbar();
        self.sync_page_selection();
    }

    pub fn reveal_page(self: &Rc<Self>, page: usize) {
        {
            let mut state = self.state.borrow_mut();
            if state.doc.page_range(page).is_none() {
                return;
            }
            state.page_forward = page >= state.selected_page;
            state.selected_page = page;
            state.highlighted = None;
        }
        self.show_page(false);
        self.refresh_toolbar();
        self.sync_page_selection();
    }

    fn set_mode(self: &Rc<Self>, mode: Mode) {
        {
            let mut state = self.state.borrow_mut();
            if state.mode == mode {
                return;
            }
            state.mode = mode;
            state.settings.reading_mode = mode.key().into();
            state.settings.save();
        }
        self.show_page(false);
        self.refresh_toolbar();
    }

    fn change_font(self: &Rc<Self>, step: f64) {
        {
            let mut state = self.state.borrow_mut();
            if state.mode == Mode::Original {
                return;
            }
            let size = if step == 0.0 { DEFAULT_FONT_SIZE } else { (state.settings.font_size + step).clamp(MIN_FONT_SIZE, MAX_FONT_SIZE) };
            state.settings.font_size = size;
            state.settings.save();
        }
        self.apply_font();
        self.refresh_toolbar();
    }

    fn toggle_assistant(self: &Rc<Self>) {
        let show = !self.state.borrow().settings.show_assistant;
        self.show_assistant(show);
    }

    pub fn show_assistant(self: &Rc<Self>, show: bool) {
        {
            let mut state = self.state.borrow_mut();
            if state.settings.show_assistant != show {
                state.settings.show_assistant = show;
                state.settings.save();
            }
        }
        self.updating.set(true);
        self.ui.assistant_toggle.set_active(show);
        self.updating.set(false);
        self.ui.inner.set_show_sidebar(show);
        if show {
            self.chat_prepare();
        }
    }

    /// The paragraph's "ask" action: explain this passage.
    fn ask_about(self: &Rc<Self>, index: usize) {
        let question = {
            let state = self.state.borrow();
            let Some(paragraph) = state.doc.paragraphs().get(index) else { return };
            format!("Explique em linguagem simples este trecho da página {}: «{}»", paragraph.page, prefix(&paragraph.original, 280))
        };
        self.show_assistant(true);
        self.chat_ask(&question, Some(false));
    }

    fn open_in_pdf(self: &Rc<Self>, index: usize) {
        {
            let mut state = self.state.borrow_mut();
            state.mode = Mode::Original;
            state.settings.reading_mode = Mode::Original.key().into();
            state.settings.save();
        }
        self.reveal_paragraphs(index..index + 1);
    }

    /// The original page, with the paragraph marked, shown over the reading view. Esc returns to reading.
    fn peek(self: &Rc<Self>, index: usize, anchor: &gtk::Button) {
        let (document, page, bounds) = {
            let state = self.state.borrow();
            let Some(paragraph) = state.doc.paragraphs().get(index) else { return };
            (state.pdf.clone(), paragraph.page, paragraph.bounds)
        };
        let Some(document) = document else { return self.open_in_pdf(index) };
        let popover = gtk::Popover::builder().position(gtk::PositionType::Left).build();
        let title = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(2).hexpand(true).build();
        title.append(&text_label(&format!("Página {page} · PDF original"), &["heading"]));
        title.append(&text_label(&if bounds.is_some() { format!("Parágrafo {} marcado", index + 1) } else { "Trecho sem posição no PDF".to_string() }, &["caption", "dim-label"]));
        let full = gtk::Button::builder().label("Abrir no modo PDF").tooltip_text("Trocar a leitura para o PDF original neste trecho").build();
        let close = icon_button("window-close-symbolic", "Fechar (Esc)");
        let head = gtk::Box::builder().orientation(gtk::Orientation::Horizontal).spacing(10).margin_top(6).margin_bottom(8).margin_start(6).margin_end(6).build();
        head.append(&title);
        head.append(&full);
        head.append(&close);
        let view = pdf::view(&document, page, bounds);
        view.set_size_request(540, 600);
        let content = gtk::Box::new(gtk::Orientation::Vertical, 0);
        content.append(&head);
        content.append(&gtk::Separator::new(gtk::Orientation::Horizontal));
        content.append(&view);
        popover.set_child(Some(&content));
        popover.set_parent(anchor);
        let closing = popover.clone();
        close.connect_clicked(move |_| closing.popdown());
        let (app, closing) = (self.clone(), popover.clone());
        full.connect_clicked(move |_| {
            closing.popdown();
            app.open_in_pdf(index);
        });
        popover.connect_closed(|popover| {
            let popover = popover.clone();
            glib::idle_add_local_once(move || popover.unparent());
        });
        popover.popup();
    }

    // Export

    fn export(self: &Rc<Self>, format: ExportFormat) {
        let name = {
            let state = self.state.borrow();
            if state.doc.completed() == 0 || state.phase != Phase::Idle {
                return;
            }
            format!("{} — tradução.{}", exporter::stem(&state.filename), format.extension())
        };
        let dialog = gtk::FileDialog::builder().title(format.title()).modal(true).initial_name(name).build();
        let app = self.clone();
        dialog.save(Some(&self.window), gio::Cancellable::NONE, move |result| {
            let Some(path) = result.ok().and_then(|file| file.path()) else { return };
            let outcome = {
                let state = app.state.borrow();
                let target = language_name(&state.target);
                match format {
                    ExportFormat::Pdf => pdf::export(&path, &state.filename, &target, &state.doc),
                    ExportFormat::Markdown => std::fs::write(&path, exporter::markdown(&state.filename, &target, &state.doc)).map_err(|error| error.to_string()),
                    ExportFormat::Text => std::fs::write(&path, exporter::text(&state.filename, &target, &state.doc)).map_err(|error| error.to_string()),
                }
            };
            match outcome {
                Ok(()) => {
                    app.state.borrow_mut().status = format!("Exportado para {}.", file_name(&path));
                    app.refresh_sidebar();
                }
                Err(error) => app.alert(&error),
            }
        });
    }

    fn persist(&self) {
        {
            let state = self.state.borrow();
            if state.fingerprint.is_empty() || state.doc.is_empty() {
                return;
            }
            let path = state.path.as_ref().map(|path| path.display().to_string()).unwrap_or_default();
            let mut saved = self.library.load(&state.fingerprint).unwrap_or_else(|| SavedDocument::new(&state.filename, &path, state.page_count, &state.source, &state.target, Vec::new()));
            saved.name = state.filename.clone();
            if !path.is_empty() {
                saved.path = path;
            }
            saved.pages = state.page_count;
            saved.source = state.source.clone();
            saved.target = state.target.clone();
            saved.translations = state.doc.paragraphs().iter().map(|paragraph| paragraph.translated.clone()).collect();
            saved.content_key = Some(state.content_key.clone());
            saved.opened = now();
            self.library.save(&saved, &state.fingerprint);
        }
        self.state.borrow_mut().recents = self.library.recents();
    }

    pub fn alert(&self, message: &str) {
        let dialog = adw::AlertDialog::new(Some("Não foi possível concluir"), Some(message));
        dialog.add_response("ok", "OK");
        dialog.present(Some(&self.window));
    }

    // Keeping the widgets in step with the state

    fn refresh(self: &Rc<Self>) {
        self.refresh_start();
        self.refresh_sidebar();
        self.refresh_toolbar();
    }

    fn refresh_start(self: &Rc<Self>) {
        let state = self.state.borrow();
        let ui = &self.ui;
        let importing = state.phase == Phase::Importing;
        ui.start_progress_box.set_visible(importing);
        ui.start_progress.set_fraction(state.import_progress);
        ui.start_status.set_text(&state.status);
        ui.start_cancel.set_sensitive(!state.stopping);
        ui.drop_zone.set_visible(!importing);
        ui.recents_box.set_visible(!importing && !state.recents.is_empty());
        ui.recents_list.remove_all();
        for recent in state.recents.iter().take(5) {
            let percent = (recent.completed * 100).checked_div(recent.paragraphs).unwrap_or(0);
            let progress = match percent {
                0 => "não traduzido".to_string(),
                100 => "tradução concluída".to_string(),
                _ => format!("{percent}% traduzido"),
            };
            let name = gtk::Label::builder().label(&recent.name).xalign(0.0).ellipsize(pango::EllipsizeMode::Middle).tooltip_text(&recent.name).build();
            let detail = text_label(&format!("{} páginas · {progress}", recent.pages), &["caption", "dim-label"]);
            let texts = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(2).hexpand(true).build();
            texts.append(&name);
            texts.append(&detail);
            let when = text_label(&relative_time(recent.opened), &["caption", "dim-label"]);
            let remove = icon_button("user-trash-symbolic", "Remover dos recentes");
            remove.add_css_class("flat");
            remove.set_valign(gtk::Align::Center);
            let (app, id) = (self.clone(), recent.id.clone());
            remove.connect_clicked(move |_| app.forget(&id));
            let icon = gtk::Image::from_icon_name("text-x-generic-symbolic");
            icon.add_css_class("accent");
            let row = gtk::Box::builder().orientation(gtk::Orientation::Horizontal).spacing(12).margin_top(8).margin_bottom(8).margin_start(12).margin_end(8).build();
            row.append(&icon);
            row.append(&texts);
            row.append(&when);
            row.append(&remove);
            ui.recents_list.append(&gtk::ListBoxRow::builder().child(&row).activatable(true).build());
        }
    }

    fn refresh_sidebar(&self) {
        let state = self.state.borrow();
        let ui = &self.ui;
        let busy = state.phase != Phase::Idle;
        ui.file_label.set_text(&state.filename);
        ui.file_label.set_tooltip_text(Some(&state.filename));
        ui.counts_label.set_text(&format!("{} páginas · {} parágrafos", state.page_count, state.doc.count()));
        let fraction = state.doc.completed() as f64 / state.doc.count().max(1) as f64;
        ui.progress.set_fraction(fraction);
        ui.progress_label.set_text(&format!("{} de {} parágrafos", state.doc.completed(), state.doc.count()));
        ui.percent_label.set_text(&format!("{}%", (fraction * 100.0) as u32));
        ui.source.set_sensitive(!busy);
        ui.target.set_sensitive(!busy);
        // Done is a state, not a disabled button: a disabled control reads as "unavailable".
        let complete = state.doc.is_complete();
        ui.done_label.set_visible(complete);
        ui.translate.set_visible(!complete);
        if state.phase == Phase::Translating {
            ui.translate.set_label(if state.stopping { "Aguarde…" } else { "Pausar" });
            ui.translate.set_sensitive(!state.stopping);
            ui.translate.remove_css_class("suggested-action");
        } else {
            ui.translate.set_label(if state.doc.completed() > 0 { "Continuar tradução" } else { "Traduzir documento" });
            ui.translate.set_sensitive(!busy);
            ui.translate.add_css_class("suggested-action");
        }
        let importing = state.phase == Phase::Importing;
        ui.import_progress.set_visible(importing);
        ui.import_progress.set_fraction(state.import_progress);
        ui.import_cancel.set_visible(importing);
        ui.import_cancel.set_sensitive(!state.stopping);
        ui.spinner.set_visible(busy);
        ui.spinner.set_spinning(busy);
        ui.status.set_text(&state.status);
        if let Some(button) = &self.page.borrow().translate_here {
            let pending = state.doc.completed_on(state.selected_page) < state.doc.page_range(state.selected_page).map_or(0, |range| range.len());
            button.set_visible(pending && state.phase != Phase::Translating);
            button.set_sensitive(!busy);
        }
    }

    fn refresh_toolbar(&self) {
        let state = self.state.borrow();
        let ui = &self.ui;
        let busy = state.phase != Phase::Idle;
        let text_mode = state.mode != Mode::Original;
        ui.previous.set_sensitive(state.doc.pages().first() != Some(&state.selected_page) && !state.doc.is_empty());
        ui.next.set_sensitive(state.doc.pages().last() != Some(&state.selected_page) && !state.doc.is_empty());
        ui.smaller.set_sensitive(text_mode && state.settings.font_size > MIN_FONT_SIZE);
        ui.larger.set_sensitive(text_mode && state.settings.font_size < MAX_FONT_SIZE);
        ui.export.set_sensitive(state.doc.completed() > 0 && !busy);
        ui.open.set_sensitive(!busy);
        ui.home.set_sensitive(!busy);
        ui.choose.set_sensitive(!busy);
        self.updating.set(true);
        for (mode, button) in Mode::ALL.into_iter().zip(&ui.modes) {
            button.set_active(mode == state.mode);
        }
        self.updating.set(false);
        self.window.set_title(Some(if state.doc.is_empty() { "Parágrafo" } else { &state.filename }));
    }

    /// The source list starts with automatic detection; each language appears once.
    fn refresh_languages(&self) {
        let state = self.state.borrow();
        let automatic = state.detected.as_ref().map_or("Detectar automaticamente".to_string(), |code| format!("Automático · {}", language_name(code)));
        let names: Vec<&str> = LANGUAGES.iter().map(|language| language.name).collect();
        let mut sources = vec![automatic.as_str()];
        sources.extend(&names);
        let position = |code: &str| LANGUAGES.iter().position(|language| language::same_language(language.code, code));
        self.updating.set(true);
        self.ui.source.set_model(Some(&gtk::StringList::new(&sources)));
        self.ui.source.set_selected(if state.source == "auto" { 0 } else { position(&state.source).map_or(0, |at| at as u32 + 1) });
        self.ui.target.set_model(Some(&gtk::StringList::new(&names)));
        self.ui.target.set_selected(position(&state.target).unwrap_or(0) as u32);
        self.updating.set(false);
    }

    fn page_status(state: &State, page: usize) -> (String, String) {
        let range = state.doc.page_range(page).unwrap_or(0..0);
        let snippet = state.doc.paragraphs().get(range.start).map_or(String::new(), |paragraph| {
            prefix(if paragraph.translated.is_empty() { &paragraph.original } else { &paragraph.translated }, 80).replace('\n', " ")
        });
        let (done, total) = (state.doc.completed_on(page), range.len());
        let progress = if total > 0 && done >= total {
            "✓".to_string()
        } else if done > 0 {
            format!("{done}/{total}")
        } else {
            String::new()
        };
        (snippet, progress)
    }

    fn rebuild_pages(self: &Rc<Self>) {
        self.updating.set(true);
        self.ui.pages.remove_all();
        let mut rows = HashMap::new();
        {
            let state = self.state.borrow();
            for &page in state.doc.pages() {
                let (snippet_text, progress_text) = Self::page_status(&state, page);
                let title = text_label(&format!("Página {page}"), &[]);
                let snippet = gtk::Label::builder().label(&snippet_text).xalign(0.0).ellipsize(pango::EllipsizeMode::End).build();
                snippet.add_css_class("caption");
                snippet.add_css_class("dim-label");
                let texts = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(2).hexpand(true).build();
                texts.append(&title);
                texts.append(&snippet);
                let progress = gtk::Label::new(Some(&progress_text));
                progress.add_css_class("page-status");
                progress.add_css_class("accent");
                let row = gtk::Box::builder().orientation(gtk::Orientation::Horizontal).spacing(8).margin_top(2).margin_bottom(2).build();
                row.append(&texts);
                row.append(&progress);
                self.ui.pages.append(&row);
                rows.insert(page, (snippet, progress));
            }
        }
        self.page.borrow_mut().sidebar = rows;
        self.updating.set(false);
        self.sync_page_selection();
    }

    fn sync_page_selection(&self) {
        let position = {
            let state = self.state.borrow();
            state.doc.pages().iter().position(|&page| page == state.selected_page)
        };
        self.updating.set(true);
        if let Some(row) = position.and_then(|at| self.ui.pages.row_at_index(at as i32)) {
            self.ui.pages.select_row(Some(&row));
        }
        self.updating.set(false);
    }

    // The reading page

    fn show_page(self: &Rc<Self>, animate: bool) {
        let (child, forward) = {
            let state = self.state.borrow();
            if state.doc.is_empty() {
                return;
            }
            let child: gtk::Widget = if state.mode == Mode::Original { self.pdf_page(&state) } else { self.text_page(&state) };
            (child, state.page_forward)
        };
        let stack = &self.ui.reading;
        let animate = animate && gtk::Settings::default().is_some_and(|settings| settings.is_gtk_enable_animations());
        stack.set_transition_type(match (animate, forward) {
            (false, _) => gtk::StackTransitionType::None,
            (true, true) => gtk::StackTransitionType::SlideLeft,
            (true, false) => gtk::StackTransitionType::SlideRight,
        });
        let old = stack.visible_child();
        stack.add_child(&child);
        stack.set_visible_child(&child);
        if let Some(old) = old {
            let stack = stack.clone();
            let remove = move || {
                if old.parent().is_some() {
                    stack.remove(&old);
                }
            };
            if animate {
                glib::timeout_add_local_once(Duration::from_millis(320), remove);
            } else {
                remove();
            }
        }
        self.scroll_to_highlight();
        self.refresh_sidebar();
    }

    fn highlight_bounds(state: &State) -> Option<Rect> {
        let range = state.highlighted.clone()?;
        union_all(range.filter_map(|index| state.doc.paragraphs().get(index)).filter(|paragraph| paragraph.page == state.selected_page).filter_map(|paragraph| paragraph.bounds))
    }

    fn pdf_page(&self, state: &State) -> gtk::Widget {
        let mut page = self.page.borrow_mut();
        page.rows.clear();
        page.scroll = None;
        page.translate_here = None;
        match &state.pdf {
            Some(document) => pdf::view(document, state.selected_page, Self::highlight_bounds(state)).upcast(),
            None => adw::StatusPage::builder().icon_name("dialog-warning-symbolic").title("PDF original indisponível").description("O arquivo não pôde ser aberto para exibição.").build().upcast(),
        }
    }

    fn text_page(self: &Rc<Self>, state: &State) -> gtk::Widget {
        let translation_only = state.mode == Mode::Translation;
        let range = state.doc.page_range(state.selected_page).unwrap_or(0..0);
        let column = gtk::Box::new(gtk::Orientation::Vertical, 0);
        column.add_css_class("reading-page");

        let title = gtk::Label::new(Some(&format!("Página {}", state.selected_page)));
        title.add_css_class("page-title");
        let of = gtk::Label::builder().label(format!("de {}", state.page_count)).hexpand(true).xalign(0.0).valign(gtk::Align::Baseline).build();
        of.add_css_class("dim-label");
        let here = gtk::Button::builder().label("Traduzir a partir daqui").valign(gtk::Align::Center).build();
        let app = self.clone();
        here.connect_clicked(move |_| app.start_translation());
        let head = gtk::Box::builder().orientation(gtk::Orientation::Horizontal).spacing(8).margin_start(12).margin_end(12).margin_bottom(10).build();
        head.append(&title);
        head.append(&of);
        head.append(&here);
        column.append(&head);
        if !state.notice.is_empty() {
            let notice = text_label(&state.notice, &["caption", "dim-label"]);
            notice.set_margin_start(12);
            notice.set_margin_bottom(10);
            column.append(&notice);
        }
        if !translation_only {
            let origin = if state.source == "auto" { state.detected.clone().unwrap_or_default() } else { state.source.clone() };
            let original = section_label(&format!("Original · {}", language_name(&origin)));
            let translated = text_label(&language_name(&state.target).to_uppercase(), &["section-label", "accent"]);
            let labels = gtk::Box::builder().orientation(gtk::Orientation::Horizontal).spacing(36).homogeneous(true).margin_start(50).margin_end(116).margin_bottom(6).build();
            labels.append(&original);
            labels.append(&translated);
            column.append(&labels);
        }

        let mut rows = HashMap::new();
        for index in range.clone() {
            let row = self.paragraph_row(state, index, translation_only);
            column.append(&row.container);
            if index + 1 != range.end {
                column.append(&gtk::Separator::builder().orientation(gtk::Orientation::Horizontal).margin_start(46).build());
            }
            rows.insert(index, row);
        }

        let previous = gtk::Button::builder().label("Página anterior").tooltip_text("Também com ← ou deslizando dois dedos para a direita").build();
        previous.set_sensitive(state.doc.pages().first() != Some(&state.selected_page));
        let next = gtk::Button::builder().label("Próxima página").tooltip_text("Também com → ou deslizando dois dedos para a esquerda").build();
        next.set_sensitive(state.doc.pages().last() != Some(&state.selected_page));
        for (button, offset) in [(&previous, -1), (&next, 1)] {
            button.add_css_class("pill");
            let app = self.clone();
            button.connect_clicked(move |_| app.move_page(offset));
        }
        let hint = gtk::Label::builder().label("← → mudam de página").hexpand(true).build();
        hint.add_css_class("caption");
        hint.add_css_class("dim-label");
        let pager = gtk::Box::builder().orientation(gtk::Orientation::Horizontal).margin_top(28).margin_start(12).margin_end(12).build();
        pager.append(&previous);
        pager.append(&hint);
        pager.append(&next);
        column.append(&pager);

        let clamp = adw::Clamp::builder().maximum_size(if translation_only { 760 } else { 1180 }).child(&column).build();
        let scroll = gtk::ScrolledWindow::builder().hscrollbar_policy(gtk::PolicyType::Never).child(&clamp).build();
        scroll.add_css_class("paper");
        let mut page = self.page.borrow_mut();
        page.rows = rows;
        page.scroll = Some(scroll.clone());
        page.translate_here = Some(here);
        scroll.upcast()
    }

    fn reading_label(text: &str, classes: &[&str]) -> gtk::Label {
        let label = gtk::Label::builder().label(text).xalign(0.0).yalign(0.0).wrap(true).wrap_mode(pango::WrapMode::WordChar).selectable(true).hexpand(true).valign(gtk::Align::Start).build();
        label.add_css_class("reading-text");
        for class in classes {
            label.add_css_class(class);
        }
        label
    }

    fn paragraph_row(self: &Rc<Self>, state: &State, index: usize, translation_only: bool) -> RowWidgets {
        let paragraph = &state.doc.paragraphs()[index];
        let container = gtk::Box::new(gtk::Orientation::Horizontal, 14);
        container.add_css_class("paragraph");
        if state.highlighted.as_ref().is_some_and(|range| range.contains(&index)) {
            container.add_css_class("highlighted");
        }
        let number = gtk::Label::builder().label((index + 1).to_string()).width_request(24).xalign(1.0).valign(gtk::Align::Start).build();
        number.add_css_class("number");
        number.add_css_class("dim-label");
        number.update_property(&[gtk::accessible::Property::Label(&format!("Parágrafo {}", index + 1))]);
        container.append(&number);

        let slot = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(6).hexpand(true).build();
        Self::fill_slot(&slot, paragraph, state.active == Some(index), translation_only);
        if translation_only {
            container.append(&slot);
        } else {
            let columns = gtk::Box::builder().orientation(gtk::Orientation::Horizontal).spacing(36).homogeneous(true).hexpand(true).build();
            columns.append(&Self::reading_label(&paragraph.original, &["original"]));
            columns.append(&slot);
            container.append(&columns);
        }

        let actions = gtk::Box::builder().orientation(gtk::Orientation::Horizontal).spacing(2).valign(gtk::Align::Start).build();
        actions.add_css_class("actions");
        let copy = icon_button("edit-copy-symbolic", "Copiar a tradução (ou o original, se ainda não traduzido)");
        let ask = icon_button("dialog-question-symbolic", "Perguntar sobre este trecho");
        let show = icon_button("find-location-symbolic", "Mostrar no PDF original");
        for button in [&copy, &ask, &show] {
            button.add_css_class("flat");
            actions.append(button);
        }
        let app = self.clone();
        copy.connect_clicked(move |button| {
            let text = app.state.borrow().doc.paragraphs().get(index).map(|p| if p.translated.is_empty() { p.original.clone() } else { p.translated.clone() });
            if let Some(text) = text {
                button.clipboard().set_text(&text);
                button.set_icon_name("object-select-symbolic");
                let button = button.clone();
                glib::timeout_add_local_once(Duration::from_millis(1400), move || button.set_icon_name("edit-copy-symbolic"));
            }
        });
        let app = self.clone();
        ask.connect_clicked(move |_| app.ask_about(index));
        let app = self.clone();
        show.connect_clicked(move |button| app.peek(index, button));
        container.append(&actions);
        RowWidgets { container, slot }
    }

    fn fill_slot(slot: &gtk::Box, paragraph: &extraction::Paragraph, active: bool, translation_only: bool) {
        clear(slot);
        if !paragraph.translated.is_empty() {
            slot.append(&Self::reading_label(&paragraph.translated, &[]));
            return;
        }
        if translation_only {
            // Nothing to show yet: keep the original readable instead of a blank.
            slot.append(&Self::reading_label(&paragraph.original, &["pending"]));
        }
        let waiting = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        if active {
            let spinner = gtk::Spinner::new();
            spinner.set_spinning(true);
            waiting.append(&spinner);
        }
        let caption = match (active, translation_only) {
            (true, _) => "Traduzindo…",
            (false, true) => "Ainda no idioma original",
            (false, false) => "Tradução pendente",
        };
        waiting.append(&text_label(caption, &["caption", "dim-label"]));
        slot.append(&waiting);
    }

    /// One paragraph changed (translated, or being translated now): refresh its row and its page in the sidebar.
    fn update_row(&self, index: usize) {
        let state = self.state.borrow();
        let Some(paragraph) = state.doc.paragraphs().get(index) else { return };
        let page = self.page.borrow();
        if let Some(row) = page.rows.get(&index) {
            Self::fill_slot(&row.slot, paragraph, state.active == Some(index), state.mode == Mode::Translation);
        }
        if let Some((snippet, progress)) = page.sidebar.get(&paragraph.page) {
            let (snippet_text, progress_text) = Self::page_status(&state, paragraph.page);
            snippet.set_text(&snippet_text);
            progress.set_text(&progress_text);
        }
    }

    fn scroll_to_highlight(&self) {
        let target = {
            let state = self.state.borrow();
            let page = self.page.borrow();
            state.highlighted.as_ref().and_then(|range| page.rows.get(&range.start)).map(|row| row.container.clone()).zip(page.scroll.clone())
        };
        let Some((row, scroll)) = target else { return };
        glib::timeout_add_local_once(Duration::from_millis(340), move || {
            let Some(content) = scroll.child() else { return };
            if let Some(point) = row.compute_point(&content, &gtk::graphene::Point::new(0.0, 0.0)) {
                scroll.vadjustment().set_value((f64::from(point.y()) - 24.0).max(0.0));
            }
        });
    }
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()
}

fn is_pdf(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("pdf"))
}

fn main() -> glib::ExitCode {
    let application = adw::Application::builder().application_id(APP_ID).flags(gio::ApplicationFlags::HANDLES_OPEN).build();
    application.connect_startup(|_| gtk::Window::set_default_icon_name(APP_ID));
    // One window: a PDF opened from the file manager goes to it instead of spawning a second one.
    let instance: Rc<RefCell<Option<Rc<App>>>> = Rc::new(RefCell::new(None));
    let present = move |application: &adw::Application| -> Rc<App> {
        let app = instance.borrow_mut().get_or_insert_with(|| App::new(application)).clone();
        app.window.present();
        app
    };
    let activate = present.clone();
    application.connect_activate(move |application| {
        activate(application);
    });
    application.connect_open(move |application, files, _| {
        let app = present(application);
        if let Some(path) = files.iter().filter_map(|file| file.path()).find(|path| is_pdf(path)) {
            app.load(path);
        }
    });
    application.run()
}
