//! The "ask the document" side panel: availability of Ollama, the search index, questions and the synthesis.

use crate::{clear, section_label, text_label, App, Event};
use adw::prelude::*;
use gtk::{glib, pango};
use paragrafo_core::answer::{self, Destination};
use paragrafo_core::assistant::{self, Availability, ChatMessage, CheckResult, IndexState, Overview, Question, Role, Stage, Update};
use paragrafo_core::extraction::Paragraph;
use paragrafo_core::ollama::{OllamaError, OllamaMessage};
use paragrafo_core::rag::{self, Chunk, Retrieved, VectorIndex};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub enum ChatEvent {
    Checked(CheckResult),
    IndexProgress { generation: u64, done: usize, total: usize },
    Indexed { generation: u64, result: Result<Arc<VectorIndex>, OllamaError> },
    Pull { model: String, progress: f64 },
    PullEnded(Result<(), OllamaError>),
    Answer { run: u64, id: u64, update: Update },
    AnswerEnded { run: u64, id: u64, result: Result<(), OllamaError> },
}

/// A question waiting for the search index to be ready.
struct Pending {
    id: u64,
    question: String,
    previous: Option<String>,
    history: Vec<OllamaMessage>,
    translations: Vec<String>,
}

pub struct ChatState {
    availability: Availability,
    index_state: IndexState,
    messages: Vec<ChatMessage>,
    chat_models: Vec<String>,
    pub chat_model: String,
    busy: bool,
    checking: bool,
    indexing: bool,
    fingerprint: String,
    chunks: Arc<Vec<Chunk>>,
    paragraphs: Arc<Vec<Paragraph>>,
    index: Option<Arc<VectorIndex>>,
    /// Changes with the document, so results for a previous one are dropped.
    generation: u64,
    index_cancel: Arc<AtomicBool>,
    /// Changes with every answer (and when one is stopped), so late pieces are dropped.
    run: u64,
    answer_cancel: Arc<AtomicBool>,
    pull_cancel: Arc<AtomicBool>,
    next_id: u64,
    pending: Option<Pending>,
    expanded: HashSet<u64>,
}

impl ChatState {
    pub fn new(chat_model: &str) -> Self {
        Self {
            availability: Availability::Unknown,
            index_state: IndexState::Empty,
            messages: Vec::new(),
            chat_models: Vec::new(),
            chat_model: chat_model.to_string(),
            busy: false,
            checking: false,
            indexing: false,
            fingerprint: String::new(),
            chunks: Arc::new(Vec::new()),
            paragraphs: Arc::new(Vec::new()),
            index: None,
            generation: 0,
            index_cancel: Arc::new(AtomicBool::new(false)),
            run: 0,
            answer_cancel: Arc::new(AtomicBool::new(false)),
            pull_cancel: Arc::new(AtomicBool::new(false)),
            next_id: 1,
            pending: None,
            expanded: HashSet::new(),
        }
    }

    fn new_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn message(&mut self, id: u64) -> Option<&mut ChatMessage> {
        self.messages.iter_mut().rev().find(|message| message.id == id)
    }
}

pub struct ChatUi {
    status: gtk::Label,
    menu: gtk::MenuButton,
    menu_box: gtk::Box,
    menu_signature: RefCell<String>,
    stack: gtk::Stack,
    setup: adw::StatusPage,
    setup_actions: gtk::Box,
    setup_kind: RefCell<String>,
    pull_bar: gtk::ProgressBar,
    pull_label: gtk::Label,
    suggestions: Vec<gtk::Button>,
    conversation: gtk::Box,
    conversation_scroll: gtk::ScrolledWindow,
    entry: gtk::Entry,
    send: gtk::Button,
    rows: RefCell<Vec<(u64, gtk::Box)>>,
}

const SUGGESTIONS: [&str; 4] = ["Síntese geral do documento", "Quais são os pontos principais?", "Quais números e datas importantes aparecem?", "Onde o documento fala sobre…"];

pub fn build() -> (ChatUi, gtk::Box) {
    let title = text_label("Perguntar ao documento", &["heading"]);
    let status = text_label("Verificando o Ollama…", &["caption", "dim-label"]);
    let titles = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(3).hexpand(true).build();
    titles.append(&title);
    titles.append(&status);
    let menu_box = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(4).margin_top(4).margin_bottom(4).margin_start(4).margin_end(4).build();
    let menu = gtk::MenuButton::builder().icon_name("open-menu-symbolic").tooltip_text("Modelo e opções do assistente").valign(gtk::Align::Center).popover(&gtk::Popover::builder().child(&menu_box).build()).build();
    menu.add_css_class("flat");
    menu.update_property(&[gtk::accessible::Property::Label("Modelo e opções do assistente")]);
    let header = gtk::Box::builder().orientation(gtk::Orientation::Horizontal).spacing(10).margin_top(12).margin_bottom(12).margin_start(16).margin_end(12).build();
    header.append(&titles);
    header.append(&menu);

    let setup_actions = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(10).halign(gtk::Align::Center).build();
    let setup = adw::StatusPage::builder().child(&setup_actions).vexpand(true).build();
    setup.add_css_class("compact");
    let pull_bar = gtk::ProgressBar::builder().width_request(200).build();
    let pull_label = text_label("", &["caption", "dim-label", "numeric"]);
    pull_label.set_halign(gtk::Align::Center);

    let icon = gtk::Image::builder().icon_name("system-search-symbolic").pixel_size(26).halign(gtk::Align::Start).build();
    icon.add_css_class("accent");
    let heading = text_label("O que você quer saber?", &["chat-title"]);
    let about = text_label("As respostas vêm do próprio documento e indicam a página e o trecho de onde saíram. Clique em uma fonte para ir até ela.", &["dim-label"]);
    let suggestions_box = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(10).margin_top(16).margin_bottom(16).margin_start(16).margin_end(16).build();
    suggestions_box.append(&icon);
    suggestions_box.append(&heading);
    suggestions_box.append(&about);
    let suggestions: Vec<gtk::Button> = SUGGESTIONS
        .iter()
        .map(|text| {
            let button = gtk::Button::builder().child(&text_label(text, &[])).build();
            button.add_css_class("suggestion");
            button.add_css_class("card");
            suggestions_box.append(&button);
            button
        })
        .collect();
    let suggestions_scroll = gtk::ScrolledWindow::builder().hscrollbar_policy(gtk::PolicyType::Never).vexpand(true).child(&suggestions_box).build();

    let conversation = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(18).margin_top(16).margin_bottom(16).margin_start(16).margin_end(16).build();
    let conversation_scroll = gtk::ScrolledWindow::builder().hscrollbar_policy(gtk::PolicyType::Never).vexpand(true).child(&conversation).build();

    let stack = gtk::Stack::builder().vexpand(true).build();
    stack.add_named(&setup, Some("setup"));
    stack.add_named(&suggestions_scroll, Some("suggestions"));
    stack.add_named(&conversation_scroll, Some("conversation"));

    let entry = gtk::Entry::builder().placeholder_text("Pergunte sobre este documento…").hexpand(true).has_frame(false).build();
    entry.update_property(&[gtk::accessible::Property::Label("Pergunta sobre o documento")]);
    let send = gtk::Button::builder().icon_name("go-up-symbolic").valign(gtk::Align::Center).build();
    send.add_css_class("circular");
    send.add_css_class("suggested-action");
    let field = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    field.add_css_class("composer");
    field.append(&entry);
    field.append(&send);
    let disclaimer = gtk::Label::new(Some("Gerado no seu computador. Confira nos trechos citados."));
    disclaimer.add_css_class("caption");
    disclaimer.add_css_class("dim-label");
    let composer = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(6).margin_top(8).margin_bottom(10).margin_start(12).margin_end(12).build();
    composer.append(&field);
    composer.append(&disclaimer);

    let panel = gtk::Box::new(gtk::Orientation::Vertical, 0);
    panel.append(&header);
    panel.append(&gtk::Separator::new(gtk::Orientation::Horizontal));
    panel.append(&stack);
    panel.append(&composer);

    let ui = ChatUi {
        status,
        menu,
        menu_box,
        menu_signature: RefCell::new("-".into()),
        stack,
        setup,
        setup_actions,
        setup_kind: RefCell::new(String::new()),
        pull_bar,
        pull_label,
        suggestions,
        conversation,
        conversation_scroll,
        entry,
        send,
        rows: RefCell::new(Vec::new()),
    };
    (ui, panel)
}

impl App {
    fn chat_send_event(tx: &async_channel::Sender<Event>, event: ChatEvent) {
        let _ = tx.send_blocking(Event::Chat(event));
    }

    pub fn chat_connect(self: &Rc<Self>) {
        let ui = &self.chat_ui;
        let app = self.clone();
        ui.suggestions[0].connect_clicked(move |_| app.chat_ask(assistant::OVERVIEW_QUESTION, None));
        for position in [1, 2] {
            let app = self.clone();
            ui.suggestions[position].connect_clicked(move |_| app.chat_ask(SUGGESTIONS[position], None));
        }
        let app = self.clone();
        ui.suggestions[3].connect_clicked(move |_| {
            let entry = &app.chat_ui.entry;
            entry.set_text("Onde o documento fala sobre ");
            entry.grab_focus();
            entry.set_position(-1);
        });
        let app = self.clone();
        ui.entry.connect_activate(move |_| app.chat_send());
        let app = self.clone();
        ui.entry.connect_changed(move |_| app.chat_refresh_composer());
        let app = self.clone();
        ui.send.connect_clicked(move |_| {
            if app.chat.borrow().busy {
                app.chat_stop();
            } else {
                app.chat_send();
            }
        });
    }

    fn chat_send(self: &Rc<Self>) {
        let question = self.chat_ui.entry.text().to_string();
        if self.chat.borrow().busy || question.trim().is_empty() {
            return;
        }
        self.chat_ui.entry.set_text("");
        self.chat_ask(&question, None);
    }

    /// Point the assistant at a new document. Nothing is computed until the panel is shown.
    pub fn chat_attach(self: &Rc<Self>, paragraphs: Vec<Paragraph>, fingerprint: &str) {
        {
            let mut chat = self.chat.borrow_mut();
            chat.index_cancel.store(true, Ordering::Relaxed);
            chat.answer_cancel.store(true, Ordering::Relaxed);
            chat.index_cancel = Arc::new(AtomicBool::new(false));
            chat.generation += 1;
            chat.run += 1;
            chat.busy = false;
            chat.indexing = false;
            chat.pending = None;
            chat.messages.clear();
            chat.expanded.clear();
            chat.index = None;
            chat.index_state = IndexState::Empty;
            chat.fingerprint = fingerprint.to_string();
            chat.chunks = Arc::new(rag::chunks(&paragraphs));
            chat.paragraphs = Arc::new(paragraphs);
        }
        self.chat_render_messages();
        self.chat_refresh();
    }

    pub fn chat_cancel_all(&self) {
        let chat = self.chat.borrow();
        for flag in [&chat.index_cancel, &chat.answer_cancel, &chat.pull_cancel] {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Makes sure Ollama and the models are there, then builds (or restores) the search index.
    pub fn chat_prepare(self: &Rc<Self>) {
        enum Step {
            Check,
            Index,
        }
        let step = {
            let chat = self.chat.borrow();
            if chat.chunks.is_empty() || matches!(chat.availability, Availability::Pulling(..)) {
                None
            } else if chat.availability != Availability::Ready {
                Some(Step::Check)
            } else if chat.index.is_none() && !chat.indexing {
                Some(Step::Index)
            } else {
                None
            }
        };
        match step {
            Some(Step::Check) => self.chat_check(true),
            Some(Step::Index) => self.chat_start_indexing(),
            None => {}
        }
    }

    /// One check at a time, on its own thread.
    fn chat_check(self: &Rc<Self>, launch: bool) {
        let model = {
            let mut chat = self.chat.borrow_mut();
            if chat.checking {
                return;
            }
            chat.checking = true;
            if !matches!(chat.availability, Availability::Pulling(..)) {
                chat.availability = Availability::Checking;
            }
            chat.chat_model.clone()
        };
        let (tx, client) = (self.tx.clone(), self.client.clone());
        std::thread::spawn(move || Self::chat_send_event(&tx, ChatEvent::Checked(assistant::check(&client, &model, launch))));
        self.chat_refresh();
    }

    fn chat_start_indexing(self: &Rc<Self>) {
        let (generation, cancel, fingerprint, chunks) = {
            let mut chat = self.chat.borrow_mut();
            chat.indexing = true;
            (chat.generation, chat.index_cancel.clone(), chat.fingerprint.clone(), chat.chunks.clone())
        };
        let (tx, client, library) = (self.tx.clone(), self.client.clone(), self.library.clone());
        std::thread::spawn(move || {
            let progress = tx.clone();
            let result = assistant::build_index(&client, &library, &fingerprint, &chunks, &cancel, |done, total| {
                Self::chat_send_event(&progress, ChatEvent::IndexProgress { generation, done, total });
            });
            Self::chat_send_event(&tx, ChatEvent::Indexed { generation, result: result.map(Arc::new) });
        });
        self.chat_refresh();
    }

    fn chat_download(self: &Rc<Self>, models: Vec<String>) {
        let cancel = {
            let mut chat = self.chat.borrow_mut();
            if !matches!(chat.availability, Availability::Missing(_)) {
                return;
            }
            chat.availability = Availability::Pulling(models.first().cloned().unwrap_or_default(), 0.0);
            chat.pull_cancel = Arc::new(AtomicBool::new(false));
            chat.pull_cancel.clone()
        };
        let (tx, client) = (self.tx.clone(), self.client.clone());
        std::thread::spawn(move || {
            let mut result = Ok(());
            for model in models {
                let progress = tx.clone();
                let name = model.clone();
                result = client.pull(&model, &cancel, |value| Self::chat_send_event(&progress, ChatEvent::Pull { model: name.clone(), progress: value }));
                if result.is_err() {
                    break;
                }
            }
            Self::chat_send_event(&tx, ChatEvent::PullEnded(result));
        });
        self.chat_refresh();
    }

    // Questions

    /// `overview` forces or forbids the whole-document synthesis; None decides from the wording.
    pub fn chat_ask(self: &Rc<Self>, question: &str, overview: Option<bool>) {
        let question = question.trim().to_string();
        let translations: Vec<String> = self.state.borrow().doc.paragraphs().iter().map(|paragraph| paragraph.translated.clone()).collect();
        let (id, overview) = {
            let mut chat = self.chat.borrow_mut();
            if chat.busy || chat.chunks.is_empty() || question.is_empty() {
                return;
            }
            let history = assistant::history(&chat.messages);
            let previous = assistant::previous_question(&chat.messages);
            let user = chat.new_id();
            chat.messages.push(ChatMessage::new(user, Role::User, &question, Stage::Done, false));
            let overview = overview.unwrap_or_else(|| assistant::wants_overview(&question));
            let id = chat.new_id();
            chat.messages.push(ChatMessage::new(id, Role::Assistant, "", Stage::Searching, overview));
            chat.busy = true;
            chat.run += 1;
            chat.answer_cancel = Arc::new(AtomicBool::new(false));
            if !overview {
                chat.pending = Some(Pending { id, question: question.clone(), previous, history, translations });
            }
            (id, overview)
        };
        self.chat_render_messages();
        self.chat_refresh();
        if overview {
            self.chat_run_overview(id, question, false);
        } else {
            self.chat_dispatch_pending();
        }
    }

    /// Discards the saved synthesis and writes a new one in place.
    fn chat_regenerate(self: &Rc<Self>, id: u64) {
        {
            let mut chat = self.chat.borrow_mut();
            if chat.busy {
                return;
            }
            let Some(message) = chat.message(id).filter(|message| message.is_overview) else { return };
            message.text.clear();
            message.note = None;
            message.sources.clear();
            message.cited.clear();
            message.stage = Stage::Searching;
            chat.busy = true;
            chat.run += 1;
            chat.answer_cancel = Arc::new(AtomicBool::new(false));
        }
        self.chat_render_message(id);
        self.chat_refresh();
        self.chat_run_overview(id, assistant::OVERVIEW_QUESTION.to_string(), true);
    }

    /// Runs the waiting question once the index exists; until then, gets the index going.
    fn chat_dispatch_pending(self: &Rc<Self>) {
        let ready = {
            let mut chat = self.chat.borrow_mut();
            match (chat.pending.is_some(), chat.index.clone()) {
                (false, _) => return,
                (true, Some(index)) => chat.pending.take().map(|pending| (pending, index, chat.run, chat.answer_cancel.clone(), chat.chat_model.clone())),
                (true, None) => None,
            }
        };
        let Some((pending, index, run, cancel, model)) = ready else {
            if matches!(self.chat.borrow().availability, Availability::Pulling(..)) {
                return self.chat_fail_pending("Os modelos ainda estão sendo baixados. Tente de novo quando o download terminar.".into());
            }
            self.chat_prepare();
            let stuck = {
                let chat = self.chat.borrow();
                !chat.checking && !chat.indexing && chat.index.is_none()
            };
            if stuck {
                self.chat_fail_pending("O índice do documento ainda não está pronto. Tente novamente em instantes.".into());
            }
            return;
        };
        let (tx, client) = (self.tx.clone(), self.client.clone());
        std::thread::spawn(move || {
            let id = pending.id;
            let question = Question { text: &pending.question, previous: pending.previous.as_deref(), history: &pending.history, translations: &pending.translations };
            let updates = tx.clone();
            let result = assistant::answer(&client, &index, &model, &question, &cancel, &mut |update| Self::chat_send_event(&updates, ChatEvent::Answer { run, id, update }));
            Self::chat_send_event(&tx, ChatEvent::AnswerEnded { run, id, result });
        });
    }

    fn chat_fail_pending(self: &Rc<Self>, reason: String) {
        let id = {
            let mut chat = self.chat.borrow_mut();
            let Some(pending) = chat.pending.take() else { return };
            if let Some(message) = chat.message(pending.id) {
                message.text = reason;
                message.stage = Stage::Failed;
            }
            chat.busy = false;
            pending.id
        };
        self.chat_render_message(id);
        self.chat_refresh();
    }

    fn chat_run_overview(self: &Rc<Self>, id: u64, question: String, force: bool) {
        let (run, cancel, mut model, ready, fingerprint, paragraphs, chunks) = {
            let chat = self.chat.borrow();
            (chat.run, chat.answer_cancel.clone(), chat.chat_model.clone(), chat.availability == Availability::Ready, chat.fingerprint.clone(), chat.paragraphs.clone(), chat.chunks.clone())
        };
        let (tx, client, library) = (self.tx.clone(), self.client.clone(), self.library.clone());
        std::thread::spawn(move || {
            let saved = !force && question == assistant::OVERVIEW_QUESTION && library.load(&fingerprint).and_then(|saved| saved.summary).is_some_and(|text| !text.is_empty());
            let mut result = Ok(());
            // A saved synthesis needs no model; anything else needs the server and a chat model.
            if !saved && !ready {
                let checked = assistant::check(&client, &model, true);
                model = checked.chat_model.clone();
                if checked.availability == Availability::ServerOff {
                    result = Err(OllamaError::NotRunning);
                } else if checked.chat_models.is_empty() {
                    result = Err(OllamaError::ModelMissing(assistant::DEFAULT_CHAT_MODEL.into()));
                }
                Self::chat_send_event(&tx, ChatEvent::Checked(checked));
            }
            if result.is_ok() {
                let request = Overview { fingerprint: &fingerprint, paragraphs: &paragraphs, chunks: &chunks, chat_model: &model, question: &question, force, single_pass_limit: assistant::SINGLE_PASS_LIMIT };
                let updates = tx.clone();
                result = assistant::overview(&client, &library, &request, &cancel, &mut |update| Self::chat_send_event(&updates, ChatEvent::Answer { run, id, update }));
            }
            Self::chat_send_event(&tx, ChatEvent::AnswerEnded { run, id, result });
        });
    }

    /// Takes effect at once; whatever the model still sends for this answer is dropped.
    fn chat_stop(self: &Rc<Self>) {
        let id = {
            let mut chat = self.chat.borrow_mut();
            if !chat.busy {
                return;
            }
            chat.answer_cancel.store(true, Ordering::Relaxed);
            chat.run += 1;
            chat.pending = None;
            chat.busy = false;
            chat.messages.iter_mut().rev().find(|message| message.role == Role::Assistant && !matches!(message.stage, Stage::Done | Stage::Failed)).map(|message| {
                if message.text.is_empty() {
                    message.text = "Resposta interrompida.".into();
                }
                message.stage = Stage::Done;
                message.id
            })
        };
        if let Some(id) = id {
            self.chat_render_message(id);
        }
        self.chat_refresh();
    }

    fn chat_clear(self: &Rc<Self>) {
        {
            let mut chat = self.chat.borrow_mut();
            if chat.busy {
                return;
            }
            chat.messages.clear();
            chat.expanded.clear();
        }
        self.chat_render_messages();
        self.chat_refresh();
    }

    // Events from worker threads

    pub fn chat_handle(self: &Rc<Self>, event: ChatEvent) {
        match event {
            ChatEvent::Checked(result) => {
                let (model_changed, ready) = {
                    let mut chat = self.chat.borrow_mut();
                    chat.checking = false;
                    if matches!(chat.availability, Availability::Pulling(..)) {
                        return;
                    }
                    chat.availability = result.availability;
                    let changed = chat.chat_model != result.chat_model && chat.availability != Availability::ServerOff;
                    if chat.availability != Availability::ServerOff {
                        chat.chat_models = result.chat_models;
                        chat.chat_model = result.chat_model;
                    }
                    (changed, chat.availability == Availability::Ready)
                };
                if model_changed {
                    self.chat_save_model();
                }
                if ready {
                    self.chat_prepare();
                    self.chat_dispatch_pending();
                } else {
                    let reason = match &self.chat.borrow().availability {
                        Availability::Missing(models) => OllamaError::ModelMissing(models.join(", ")),
                        _ => OllamaError::NotRunning,
                    };
                    self.chat_fail_pending(reason.to_string());
                }
                self.chat_refresh();
            }
            ChatEvent::IndexProgress { generation, done, total } => {
                {
                    let mut chat = self.chat.borrow_mut();
                    if chat.generation != generation {
                        return;
                    }
                    chat.index_state = IndexState::Indexing(done, total);
                }
                self.chat_refresh();
            }
            ChatEvent::Indexed { generation, result } => {
                let failure = {
                    let mut chat = self.chat.borrow_mut();
                    if chat.generation != generation {
                        return;
                    }
                    chat.indexing = false;
                    match result {
                        Ok(index) => {
                            chat.index = Some(index);
                            chat.index_state = IndexState::Ready;
                            None
                        }
                        Err(OllamaError::Cancelled) => {
                            chat.index_state = IndexState::Empty;
                            Some(OllamaError::Cancelled.to_string())
                        }
                        Err(error) => {
                            if error == OllamaError::NotRunning {
                                chat.availability = Availability::ServerOff;
                            }
                            chat.index_state = IndexState::Failed(error.to_string());
                            Some(error.to_string())
                        }
                    }
                };
                match failure {
                    Some(reason) => self.chat_fail_pending(reason),
                    None => self.chat_dispatch_pending(),
                }
                self.chat_refresh();
            }
            ChatEvent::Pull { model, progress } => {
                self.chat.borrow_mut().availability = Availability::Pulling(model, progress);
                self.chat_refresh();
            }
            ChatEvent::PullEnded(result) => {
                {
                    let mut chat = self.chat.borrow_mut();
                    chat.availability = Availability::Unknown;
                    if let Err(error) = &result {
                        chat.index_state = IndexState::Failed(error.to_string());
                    }
                }
                match result {
                    Ok(()) => self.chat_prepare(),
                    Err(_) => self.chat_check(false),
                }
                self.chat_refresh();
            }
            ChatEvent::Answer { run, id, update } => {
                {
                    let mut chat = self.chat.borrow_mut();
                    if chat.run != run {
                        return;
                    }
                    let Some(message) = chat.message(id) else { return };
                    message.apply(update);
                }
                self.chat_render_message(id);
            }
            ChatEvent::AnswerEnded { run, id, result } => {
                {
                    let mut chat = self.chat.borrow_mut();
                    if chat.run != run {
                        return;
                    }
                    chat.busy = false;
                    let offline = result == Err(OllamaError::NotRunning);
                    if let Some(message) = chat.message(id) {
                        match result {
                            Ok(()) => message.stage = if message.stage == Stage::Failed { Stage::Failed } else { Stage::Done },
                            Err(OllamaError::Cancelled) => {
                                if message.text.is_empty() {
                                    message.text = "Resposta interrompida.".into();
                                }
                                message.stage = Stage::Done;
                            }
                            Err(error) => {
                                message.text = error.to_string();
                                message.stage = Stage::Failed;
                            }
                        }
                    }
                    if offline {
                        chat.availability = Availability::ServerOff;
                    }
                }
                self.chat_render_message(id);
                self.chat_refresh();
            }
        }
    }

    fn chat_save_model(&self) {
        let model = self.chat.borrow().chat_model.clone();
        let mut state = self.state.borrow_mut();
        state.settings.chat_model = model;
        state.settings.save();
    }

    /// A link inside an answer: go to the cited passage or page.
    fn chat_follow(self: &Rc<Self>, destination: Destination, id: u64) {
        match destination {
            Destination::Source(number) => {
                let range = self.chat.borrow().messages.iter().find(|message| message.id == id).and_then(|message| message.sources.get(number - 1)).map(|source| source.chunk.paragraphs.clone());
                if let Some(range) = range {
                    self.reveal_paragraphs(range);
                }
            }
            Destination::Page(page) => self.reveal_page(page),
        }
    }

    // Widgets

    pub fn chat_refresh(self: &Rc<Self>) {
        let chat = self.chat.borrow();
        let ui = &self.chat_ui;
        let status = match (&chat.availability, &chat.index_state) {
            (Availability::Checking | Availability::Unknown, _) => "Verificando o Ollama…".to_string(),
            (Availability::Ready, IndexState::Indexing(done, total)) => format!("Indexando {done} de {total} trechos"),
            (Availability::Ready, IndexState::Ready) => "● Pronto · roda neste computador".to_string(),
            (Availability::Ready, IndexState::Failed(_)) => "Índice indisponível".to_string(),
            (Availability::Ready, IndexState::Empty) => "Preparando…".to_string(),
            _ => "Assistente indisponível".to_string(),
        };
        ui.status.set_text(&status);
        if let (Availability::Ready, IndexState::Failed(reason)) = (&chat.availability, &chat.index_state) {
            ui.status.set_tooltip_text(Some(reason));
        } else {
            ui.status.set_tooltip_text(None);
        }

        let kind = match &chat.availability {
            Availability::ServerOff => "off".to_string(),
            Availability::Missing(models) => format!("missing:{}", models.join(",")),
            Availability::Pulling(model, _) => format!("pulling:{model}"),
            _ => String::new(),
        };
        if *ui.setup_kind.borrow() != kind {
            clear(&ui.setup_actions);
            match &chat.availability {
                Availability::ServerOff => {
                    ui.setup.set_icon_name(Some("dialog-warning-symbolic"));
                    ui.setup.set_title("O Ollama não está em execução");
                    ui.setup.set_description(Some("O assistente responde com um modelo Llama que roda no seu computador pelo Ollama. Inicie o Ollama e tente de novo."));
                    let retry = gtk::Button::with_label("Tentar novamente");
                    retry.add_css_class("suggested-action");
                    retry.add_css_class("pill");
                    let app = self.clone();
                    retry.connect_clicked(move |_| app.chat_check(true));
                    ui.setup_actions.append(&retry);
                    ui.setup_actions.append(&gtk::LinkButton::with_label("https://ollama.com/download", "Instalar o Ollama"));
                }
                Availability::Missing(models) => {
                    ui.setup.set_icon_name(Some("folder-download-symbolic"));
                    ui.setup.set_title(if models.len() == 1 { "Falta baixar um modelo" } else { "Falta baixar os modelos" });
                    ui.setup.set_description(Some(&format!("Necessário uma única vez: {}. O download fica guardado no Ollama.", models.join(" e "))));
                    let download = gtk::Button::with_label("Baixar agora");
                    download.add_css_class("suggested-action");
                    download.add_css_class("pill");
                    let (app, models) = (self.clone(), models.clone());
                    download.connect_clicked(move |_| app.chat_download(models.clone()));
                    ui.setup_actions.append(&download);
                }
                Availability::Pulling(model, _) => {
                    ui.setup.set_icon_name(Some("folder-download-symbolic"));
                    ui.setup.set_title(&format!("Baixando {model}"));
                    ui.setup.set_description(Some("Você pode continuar lendo enquanto isso."));
                    ui.setup_actions.append(&ui.pull_bar);
                    ui.setup_actions.append(&ui.pull_label);
                }
                _ => {}
            }
            ui.setup_kind.replace(kind.clone());
        }
        if let Availability::Pulling(_, progress) = &chat.availability {
            ui.pull_bar.set_fraction(*progress);
            ui.pull_label.set_text(&format!("{}%", (progress * 100.0) as u32));
        }
        ui.stack.set_visible_child_name(if !kind.is_empty() {
            "setup"
        } else if chat.messages.is_empty() {
            "suggestions"
        } else {
            "conversation"
        });

        let signature = format!("{}|{}|{}|{}", chat.chat_models.join(","), chat.chat_model, chat.messages.is_empty(), chat.busy);
        if *ui.menu_signature.borrow() != signature {
            ui.menu_signature.replace(signature);
            clear(&ui.menu_box);
            ui.menu_box.append(&section_label("Modelo de conversa"));
            let mut group: Option<gtk::CheckButton> = None;
            for model in &chat.chat_models {
                let option = gtk::CheckButton::builder().label(model).active(*model == chat.chat_model).build();
                option.set_group(group.as_ref());
                let (app, name) = (self.clone(), model.clone());
                option.connect_toggled(move |option| {
                    if option.is_active() && app.chat.borrow().chat_model != name {
                        app.chat.borrow_mut().chat_model = name.clone();
                        app.chat_save_model();
                    }
                });
                ui.menu_box.append(&option);
                group.get_or_insert(option);
            }
            if chat.chat_models.is_empty() {
                ui.menu_box.append(&text_label("Nenhum modelo instalado", &["dim-label"]));
            }
            ui.menu_box.append(&gtk::Separator::new(gtk::Orientation::Horizontal));
            let fresh = gtk::Button::with_label("Nova conversa");
            fresh.add_css_class("flat");
            fresh.set_sensitive(!chat.messages.is_empty() && !chat.busy);
            let app = self.clone();
            fresh.connect_clicked(move |_| {
                app.chat_ui.menu.popdown();
                app.chat_clear();
            });
            ui.menu_box.append(&fresh);
        }
        for button in &ui.suggestions {
            button.set_sensitive(!chat.busy && !chat.chunks.is_empty());
        }
        drop(chat);
        self.chat_refresh_composer();
    }

    fn chat_refresh_composer(&self) {
        let chat = self.chat.borrow();
        let ui = &self.chat_ui;
        if chat.busy {
            ui.send.set_icon_name("media-playback-stop-symbolic");
            ui.send.set_tooltip_text(Some("Parar resposta"));
            ui.send.update_property(&[gtk::accessible::Property::Label("Parar resposta")]);
            ui.send.set_sensitive(true);
        } else {
            ui.send.set_icon_name("go-up-symbolic");
            ui.send.set_tooltip_text(Some("Enviar pergunta"));
            ui.send.update_property(&[gtk::accessible::Property::Label("Enviar pergunta")]);
            ui.send.set_sensitive(!chat.chunks.is_empty() && !ui.entry.text().trim().is_empty());
        }
    }

    fn chat_scroll_to_end(&self) {
        let scroll = self.chat_ui.conversation_scroll.clone();
        glib::timeout_add_local_once(Duration::from_millis(40), move || {
            let adjustment = scroll.vadjustment();
            adjustment.set_value(adjustment.upper() - adjustment.page_size());
        });
    }

    fn chat_render_messages(self: &Rc<Self>) {
        let ui = &self.chat_ui;
        clear(&ui.conversation);
        let mut rows = Vec::new();
        {
            let chat = self.chat.borrow();
            for message in &chat.messages {
                let wrapper = gtk::Box::new(gtk::Orientation::Vertical, 0);
                wrapper.append(&self.chat_message_widget(message, chat.busy, chat.expanded.contains(&message.id)));
                ui.conversation.append(&wrapper);
                rows.push((message.id, wrapper));
            }
        }
        ui.rows.replace(rows);
        self.chat_scroll_to_end();
    }

    fn chat_render_message(self: &Rc<Self>, id: u64) {
        let wrapper = self.chat_ui.rows.borrow().iter().find(|(row, _)| *row == id).map(|(_, wrapper)| wrapper.clone());
        let Some(wrapper) = wrapper else { return self.chat_render_messages() };
        let widget = {
            let chat = self.chat.borrow();
            let Some(message) = chat.messages.iter().find(|message| message.id == id) else { return };
            self.chat_message_widget(message, chat.busy, chat.expanded.contains(&id))
        };
        clear(&wrapper);
        wrapper.append(&widget);
        if self.chat.borrow().messages.last().is_some_and(|message| message.id == id) {
            self.chat_scroll_to_end();
        }
    }

    fn chat_message_widget(self: &Rc<Self>, message: &ChatMessage, busy: bool, expanded: bool) -> gtk::Widget {
        if message.role == Role::User {
            let bubble = gtk::Label::builder().label(&message.text).xalign(0.0).wrap(true).wrap_mode(pango::WrapMode::WordChar).selectable(true).halign(gtk::Align::End).margin_start(36).build();
            bubble.add_css_class("bubble");
            return bubble.upcast();
        }
        let id = message.id;
        let column = gtk::Box::new(gtk::Orientation::Vertical, 12);
        let progress = match message.stage {
            Stage::Searching => Some("Procurando no documento…".to_string()),
            Stage::Reading(part, total) => Some(format!("Lendo o documento · parte {part} de {total}")),
            Stage::Writing if message.text.is_empty() => Some("Preparando a resposta…".to_string()),
            _ => None,
        };
        if let Some(progress) = progress {
            let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
            let spinner = gtk::Spinner::new();
            spinner.set_spinning(true);
            row.append(&spinner);
            row.append(&text_label(&progress, &["dim-label"]));
            column.append(&row);
        }
        if !message.text.is_empty() {
            if message.stage == Stage::Failed {
                let row = gtk::Box::new(gtk::Orientation::Horizontal, 8);
                row.append(&gtk::Image::builder().icon_name("dialog-warning-symbolic").valign(gtk::Align::Start).build());
                row.append(&text_label(&message.text, &["dim-label"]));
                column.append(&row);
            } else {
                let pages: HashSet<usize> = message.sources.iter().map(|source| source.chunk.page).collect();
                let label = gtk::Label::builder().xalign(0.0).wrap(true).wrap_mode(pango::WrapMode::WordChar).selectable(message.stage == Stage::Done).build();
                label.set_markup(&answer::pango(&message.text, message.sources.len(), Some(&pages)));
                let app = self.clone();
                label.connect_activate_link(move |_, uri| {
                    if let Some(destination) = answer::destination(uri) {
                        app.chat_follow(destination, id);
                    }
                    // Never hand a link written by the model to the system.
                    glib::Propagation::Stop
                });
                column.append(&label);
            }
        }

        // Cited passages once the answer is complete; the best matches while it is being written.
        let cited: Vec<usize> = message.cited.iter().copied().filter(|number| (1..=message.sources.len()).contains(number)).collect();
        let shown: Vec<usize> = if !cited.is_empty() { cited.clone() } else { (1..=message.sources.len().min(3)).collect() };
        if !shown.is_empty() {
            let sources = gtk::Box::new(gtk::Orientation::Vertical, 6);
            sources.append(&section_label(if cited.is_empty() { "Trechos consultados" } else { "Fontes no documento" }));
            for &number in shown.iter().take(if expanded { shown.len() } else { 3 }) {
                sources.append(&self.chat_source_row(number, &message.sources[number - 1], id));
            }
            if shown.len() > 3 {
                let toggle = gtk::Button::with_label(&if expanded { "Mostrar menos".to_string() } else { format!("Ver as {} fontes", shown.len()) });
                toggle.add_css_class("flat");
                toggle.set_halign(gtk::Align::Start);
                let app = self.clone();
                toggle.connect_clicked(move |_| {
                    {
                        let mut chat = app.chat.borrow_mut();
                        if !chat.expanded.remove(&id) {
                            chat.expanded.insert(id);
                        }
                    }
                    app.chat_render_message(id);
                });
                sources.append(&toggle);
            }
            column.append(&sources);
        }
        if let Some(note) = &message.note {
            column.append(&text_label(note, &["caption", "dim-label"]));
        }
        if message.stage == Stage::Done {
            let actions = gtk::Box::new(gtk::Orientation::Horizontal, 6);
            let copy = gtk::Button::with_label("Copiar");
            copy.add_css_class("flat");
            let text = message.text.clone();
            copy.connect_clicked(move |button| {
                button.clipboard().set_text(&text);
                button.set_label("Copiado");
                let button = button.clone();
                glib::timeout_add_local_once(Duration::from_millis(1400), move || button.set_label("Copiar"));
            });
            actions.append(&copy);
            if message.is_overview {
                let again = gtk::Button::with_label("Gerar novamente");
                again.add_css_class("flat");
                again.set_sensitive(!busy);
                let app = self.clone();
                again.connect_clicked(move |_| app.chat_regenerate(id));
                actions.append(&again);
            }
            column.append(&actions);
        }
        column.upcast()
    }

    fn chat_source_row(self: &Rc<Self>, number: usize, source: &Retrieved, id: u64) -> gtk::Button {
        let badge = gtk::Label::builder().label(number.to_string()).valign(gtk::Align::Start).build();
        badge.add_css_class("badge");
        let page = text_label(&format!("Página {}", source.chunk.page), &["caption-heading"]);
        let excerpt = gtk::Label::builder().label(source.chunk.text.replace('\n', " ")).xalign(0.0).wrap(true).wrap_mode(pango::WrapMode::WordChar).lines(2).ellipsize(pango::EllipsizeMode::End).build();
        excerpt.add_css_class("caption");
        excerpt.add_css_class("dim-label");
        let texts = gtk::Box::builder().orientation(gtk::Orientation::Vertical).spacing(2).hexpand(true).build();
        texts.append(&page);
        texts.append(&excerpt);
        let row = gtk::Box::new(gtk::Orientation::Horizontal, 10);
        row.append(&badge);
        row.append(&texts);
        row.append(&gtk::Image::builder().icon_name("go-next-symbolic").valign(gtk::Align::Start).build());
        let button = gtk::Button::builder().child(&row).tooltip_text(format!("Ir para este trecho na página {}", source.chunk.page)).build();
        button.add_css_class("source");
        button.add_css_class("card");
        button.update_property(&[gtk::accessible::Property::Label(&format!("Fonte {number}, página {}. Ir para o trecho.", source.chunk.page))]);
        let app = self.clone();
        button.connect_clicked(move |_| app.chat_follow(Destination::Source(number), id));
        button
    }
}
