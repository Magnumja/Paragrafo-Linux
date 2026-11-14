//! Porte de `Checks/main.swift` do app do Mac: agrupamento, estado, idioma, biblioteca, respostas e exportação em texto.
//! A extração de PDF real, o OCR e o PDF exportado são verificados no crate do app (`app/src/pdf.rs`).

use paragrafo_core::answer::{self, Destination};
use paragrafo_core::document::DocumentState;
use paragrafo_core::exporter;
use paragrafo_core::extraction::{self, blocks, group, Paragraph, Rect, TextLine};
use paragrafo_core::language;
use paragrafo_core::library::{Library, SavedDocument};
use paragrafo_core::translate;
use std::collections::HashSet;
use std::path::PathBuf;

fn line(text: &str, x: f64, y: f64, height: f64) -> TextLine {
    TextLine::new(text, x, y, height)
}

fn temporary(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("paragrafo-{name}-{}-{}", std::process::id(), paragrafo_core::text::now().to_bits()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn agrupamento_de_linhas() {
    assert_eq!(group(&[line("Receita de 2025:", 10., 100., 12.), line("R$ 1.250,00.", 10., 86., 12.)]), ["Receita de 2025: R$ 1.250,00."], "quebras de linha e números");
    assert_eq!(group(&[line("Primeiro.", 10., 100., 12.), line("Segundo.", 10., 65., 12.)]), ["Primeiro.", "Segundo."], "parágrafos separados");
    assert_eq!(group(&[line("日本語 العربية", 10., 100., 12.), line("• café", 10., 86., 12.)]), ["日本語 العربية", "• café"], "Unicode e listas");
    assert!(group(&[line("  ", 0., 0., 12.)]).is_empty(), "linhas vazias");
    assert_eq!(group(&[line("The trans-", 10., 100., 12.), line("lation is ready.", 10., 86., 12.)]), ["The translation is ready."], "palavra hifenizada no fim da linha é reunida");
    assert_eq!(group(&[line("Intervalo 10-", 10., 100., 12.), line("20 unidades.", 10., 86., 12.)]), ["Intervalo 10- 20 unidades."], "hífen entre números não é removido");

    let joined = blocks(&[TextLine::wide("Primeira linha", 10., 100., 12., 200.), TextLine::wide("segunda linha.", 10., 86., 12., 150.)], false);
    assert!(joined.len() == 1 && joined[0].bounds == Some(Rect::new(10., 86., 200., 26.)), "área do parágrafo é a união das linhas");

    let sized = [line("Financial results", 10., 100., 24.), line("Revenue reached 48.2 million", 10., 80., 14.), line("euros in 2025.", 10., 64., 14.)];
    let texts: Vec<String> = blocks(&sized, true).into_iter().map(|block| block.text).collect();
    assert_eq!(texts, ["Financial results", "Revenue reached 48.2 million euros in 2025."], "título em fonte maior não se funde ao parágrafo");
    assert_eq!(blocks(&sized, false).len(), 1, "OCR não separa por altura de linha");

    // Caixa de linha mais alta que o passo: a linha em branco (2 passos) fica abaixo do limite por altura.
    let tall: Vec<TextLine> = [(760.0, "Title"), (736.0, "First paragraph starts"), (724.0, "and ends here."), (700.0, "Second paragraph starts"), (688.0, "and ends."), (676.0, "Really.")]
        .iter()
        .map(|(y, text)| line(text, 16., *y, 15.))
        .collect();
    assert_eq!(blocks(&tall, false).len(), 1, "só pela altura da caixa a linha em branco passa despercebida");
    let gap = extraction::typical_gap(&tall);
    assert_eq!(gap, Some(12.0), "passo usual das linhas da página");
    let texts: Vec<String> = extraction::blocks_spaced(&tall, false, gap).into_iter().map(|block| block.text).collect();
    assert_eq!(texts, ["Title", "First paragraph starts and ends here.", "Second paragraph starts and ends. Really."], "linha em branco separa parágrafos pelo passo usual da página");
    assert_eq!(extraction::typical_gap(&tall[..3]), None, "poucas linhas não definem um passo usual");
}

#[test]
fn estado_do_documento() {
    let mut document = DocumentState::new(vec![Paragraph::new(1, "One"), Paragraph::new(1, "Two"), Paragraph::new(3, "Three")]);
    assert!(document.pages() == [1, 3] && document.page_range(1) == Some(0..2) && document.page_range(3) == Some(2..3), "navegação preserva páginas sem texto e ordem");
    document.apply("Três", 2);
    document.apply("Um", 0);
    document.apply("Um revisado", 0);
    assert!(document.completed() == 2 && document.paragraphs()[2].translated == "Três", "contagem exata com respostas fora de ordem e atualizações repetidas");
    document.apply("", 1);
    document.apply("Inválido", 99);
    assert_eq!(document.completed(), 2, "respostas vazias e índices inválidos não alteram progresso");
    document.clear_translations();
    assert!(document.completed() == 0 && document.paragraphs().iter().all(|p| p.translated.is_empty()), "trocar destino limpa todo o progresso");
    assert!(DocumentState::default().pages().is_empty() && DocumentState::default().completed() == 0, "documento vazio");

    let mut ordered = DocumentState::new(vec![Paragraph::new(1, "a"), Paragraph::new(1, "b"), Paragraph::new(2, "c"), Paragraph::new(3, "d")]);
    assert_eq!(ordered.pending_order(2), [2, 3, 0, 1], "tradução começa pela página em leitura e depois volta ao início");
    ordered.apply("C", 2);
    assert!(ordered.pending_order(2) == [3, 0, 1] && ordered.completed_on(2) == 1 && ordered.completed_on(1) == 0 && !ordered.is_complete(), "progresso por página");
    ordered.clear_translations();
    assert!(!ordered.has_page_progress(), "limpar zera o progresso por página");
}

#[test]
fn idiomas() {
    assert!(!language::needs_translation("2025 — 1.250,00 •"), "números e pontuação não exigem tradução");
    assert!(language::needs_translation("The document explains the translation process."), "texto em inglês exige tradução");
    assert!(language::needs_translation("日本語 العربية"), "texto fora do alfabeto latino exige tradução");
    let english = [Paragraph::new(1, "2025"), Paragraph::new(1, "The document explains how to translate each paragraph into another language.")];
    assert_eq!(language::detect_language(&english).as_deref(), Some("en"), "cabeçalho numérico não impede detecção do inglês");
    let portuguese = [Paragraph::new(1, "O relatório descreve como a empresa cresceu em todas as regiões durante o último ano fiscal.")];
    assert_eq!(language::detect_language(&portuguese).as_deref(), Some("pt"), "português é reconhecido");
    assert_eq!(language::detect_language(&[Paragraph::new(1, "1250")]), None, "documento numérico não recebe idioma inventado");
    assert!(language::same_language("pt", "pt-BR") && !language::same_language("en", "pt-BR"), "origem igual ao destino identificada antes da chamada");
    assert!(language::language_name("pt") == "Português" && language::language_name("xx") == "xx" && language::english_name("pt-BR") == "Portuguese", "nomes dos idiomas");
    let codes: HashSet<&str> = language::LANGUAGES.iter().map(|l| l.code).collect();
    assert_eq!(codes.len(), language::LANGUAGES.len(), "cada idioma aparece uma vez");
}

#[test]
fn trabalho_salvo() {
    let root = temporary("biblioteca");
    let library = Library::at(root.join("lib"));
    let key = "check-0123abcd";
    library.save(&SavedDocument::new("a.pdf", "/tmp/a.pdf", 3, "en", "pt", vec!["Um".into(), "".into(), "Três".into()]), key);
    library.update(key, |saved| saved.summary = Some("Síntese".into()));
    let saved = library.load(key).expect("documento salvo");
    assert!(saved.translations == ["Um", "", "Três"] && saved.summary.as_deref() == Some("Síntese") && saved.target == "pt", "tradução e síntese salvas são restauradas");
    assert!(library.recents().first().is_some_and(|r| r.id == key && r.completed == 2 && r.paragraphs == 3), "documento salvo aparece nos recentes com o progresso");
    library.save_vectors(&[vec![1., 2., 3.], vec![4., 5., 6.]], key);
    assert_eq!(library.load_vectors(key, 2, 3), Some(vec![vec![1., 2., 3.], vec![4., 5., 6.]]), "vetores do índice são restaurados");
    assert_eq!(library.load_vectors(key, 3, 3), None, "índice de tamanho diferente é recusado");
    assert_eq!(library.load_vectors(key, usize::MAX, usize::MAX), None, "dimensões malformadas não causam overflow");

    // Compatível com o que o app do Mac grava (chaves em camelCase, campos opcionais ausentes).
    let mac = r#"{"name":"m.pdf","path":"/m.pdf","pages":1,"source":"auto","target":"pt","translations":["Olá"],"contentKey":"abc","opened":752000000}"#;
    std::fs::write(library.directory().join("mac-1.json"), mac).unwrap();
    assert!(library.load("mac-1").is_some_and(|d| d.content_key.as_deref() == Some("abc") && d.index_key.is_none()), "registro no formato do Mac é lido");

    // Segurança: caminhos derivados ficam dentro da biblioteca, inclusive alvos de symlink.
    let sentinel = root.join("fora.json");
    std::fs::write(&sentinel, "preserve me").unwrap();
    library.remove("../fora");
    assert!(sentinel.exists(), "remoção recusa travessia fora da biblioteca");
    std::os::unix::fs::symlink(&sentinel, library.directory().join("symlink-test.json")).unwrap();
    library.remove("symlink-test");
    assert!(sentinel.exists() && library.load("symlink-test").is_none(), "biblioteca recusa symlink para fora da pasta");
    library.save(&SavedDocument::new("x.pdf", "/x.pdf", 1, "en", "pt", vec![]), "symlink-test");
    assert_eq!(std::fs::read_to_string(&sentinel).unwrap(), "preserve me", "gravação não segue symlink");

    library.remove(key);
    assert!(library.load(key).is_none() && !library.recents().iter().any(|r| r.id == key), "remover apaga o documento salvo");
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(library.directory()).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o700, "pasta da biblioteca é só do usuário");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn respostas_apontam_para_o_documento() {
    let linked = answer::markdown("## Resumo\n- O prazo é de 14 dias [1, 3].\n- Veja a página 12 e (p. 4) [9]", 3, None);
    assert!(linked.contains("**Resumo**") && linked.contains("• O prazo"), "títulos e listas do modelo viram texto limpo");
    assert!(linked.contains("[\\[1\\]](paragrafo://fonte/1) [\\[3\\]](paragrafo://fonte/3)") && !linked.contains("fonte/9"), "citações viram links e as inválidas somem");
    assert!(linked.contains("(paragrafo://pagina/12)") && linked.contains("(paragrafo://pagina/4)"), "menções a páginas viram links");
    let only: HashSet<usize> = [12].into_iter().collect();
    let limited = answer::markdown("Veja a página 12 e a página 4.", 0, Some(&only));
    assert!(limited.contains("pagina/12") && !limited.contains("pagina/4"), "só páginas que sustentam a resposta viram links");
    assert!(
        answer::destination("paragrafo://fonte/2") == Some(Destination::Source(2))
            && answer::destination("paragrafo://pagina/7") == Some(Destination::Page(7))
            && answer::destination("https://example.com/3").is_none(),
        "links internos são interpretados"
    );
    assert_eq!(answer::destination("paragrafo://fonte/-1"), None, "citações negativas são recusadas");
    assert_eq!(answer::plain("Prazo de **14 dias** [1].", 1, None), "Prazo de 14 dias [1].", "texto formatado preserva o conteúdo");
    assert_eq!(answer::markdown("- - Ponto duplo\n• - Outro\n* Simples", 0, None), "• Ponto duplo\n• Outro\n• Simples", "marcadores repetidos do modelo viram um só");

    let markup = answer::pango("Lucro <b>&</b> **alta** [1] na página 3.", 1, None);
    assert_eq!(
        markup,
        "Lucro &lt;b&gt;&amp;&lt;/b&gt; <b>alta</b> <a href=\"paragrafo://fonte/1\">[1]</a> na <a href=\"paragrafo://pagina/3\">página 3</a>.",
        "marcação do GTK escapa o texto do modelo e mantém os links"
    );
    assert_eq!(answer::pango("[texto](https://exemplo.com) e [solto", 0, None), "[texto](https://exemplo.com) e [solto", "links externos do modelo não viram links");
}

#[test]
fn exportacao_em_texto() {
    let mut exported = DocumentState::new(vec![Paragraph::new(1, "Hello."), Paragraph::new(2, "Pending.")]);
    exported.apply("Olá.", 0);
    let markdown = exporter::markdown("doc.pdf", "Português", &exported);
    assert!(markdown.contains("## Página 2") && markdown.contains("> Hello.") && markdown.contains("_[Não traduzido]_"), "Markdown bilíngue por página");
    let text = exporter::text("doc.pdf", "Português", &exported);
    assert!(text.starts_with("doc.pdf\nTradução para Português · 1/2 parágrafos") && text.contains("TRADUÇÃO\nOlá.") && text.contains("[Não traduzido]"), "texto bilíngue");
    assert!(exporter::stem("relatorio.final.pdf") == "relatorio.final" && exporter::stem("semextensao") == "semextensao", "nome sem extensão");
}

#[test]
fn traducao_local() {
    let installed: Vec<String> = ["llama3.2:3b", "bge-m3:latest", "translategemma:4b"].iter().map(|s| s.to_string()).collect();
    let chat: Vec<String> = vec!["llama3.2:3b".into()];
    assert_eq!(translate::pick_model(&installed, &chat, "llama3.2:3b").as_deref(), Some("translategemma:4b"), "modelo de tradução dedicado tem preferência");
    assert_eq!(translate::pick_model(&installed[..2], &chat, "llama3.2:3b").as_deref(), Some("llama3.2:3b"), "sem modelo dedicado usa o de conversa");
    assert_eq!(translate::pick_model(&[], &[], "llama3.2:3b"), None, "sem modelos não há tradução");
    let prompt = translate::prompt("en", "pt", "Hello.");
    assert!(prompt.len() == 1 && prompt[0].content.contains("English (en) to Portuguese (pt)") && prompt[0].content.ends_with("\n\n\nHello."), "prompt nomeia os idiomas e termina no texto");
    assert_eq!(translate::clean("  \"Olá, mundo.\"\n", "Hello, world."), "Olá, mundo.", "aspas acrescentadas pelo modelo são removidas");
    assert_eq!(translate::clean("“Olá”", "“Hello”"), "“Olá”", "aspas do original são mantidas");
    assert_eq!(translate::clean("```\nOlá\n```", "Hello"), "Olá", "cerca de código é removida");
}
