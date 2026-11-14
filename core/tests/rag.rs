//! Porte da parte offline de `Checks/RAG/main.swift`: divisão em trechos, busca híbrida, prompt e política de rede.

use paragrafo_core::assistant::{self, ChatMessage, Role, Stage};
use paragrafo_core::extraction::Paragraph;
use paragrafo_core::ollama::{self, ChatOptions, OllamaClient, OllamaError, OllamaMessage};
use paragrafo_core::rag::{self, Chunk, Retrieved, VectorIndex};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;

fn unit(values: &[f32]) -> Vec<f32> {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    values.iter().map(|v| v / norm).collect()
}

fn covered(chunks: &[Chunk], count: usize) -> bool {
    let indices: HashSet<usize> = chunks.iter().flat_map(|chunk| chunk.paragraphs.clone()).collect();
    let ordered = chunks.windows(2).all(|pair| pair[0].paragraphs.start <= pair[1].paragraphs.start && pair[0].paragraphs.end <= pair[1].paragraphs.end);
    indices == (0..count).collect() && ordered && chunks.iter().enumerate().all(|(offset, chunk)| offset == chunk.id)
}

fn size(text: &str) -> usize {
    text.chars().count()
}

fn long_paragraph() -> String {
    (1..=60).map(|n| format!("Sentence number {n} explains one more detail of the agreement between the parties.")).collect::<Vec<_>>().join(" ")
}

#[test]
fn chunker() {
    let sentence = "The quarterly report describes how the company performed in every region. ";
    let medium = sentence.repeat(5).trim().to_string();
    let document: Vec<Paragraph> = (1..=3).flat_map(|page| (0..6).map(move |_| page)).map(|page| Paragraph::new(page, medium.clone())).collect();
    let grouped = rag::chunks(&document);
    assert!(covered(&grouped, document.len()), "chunker cobre todos os parágrafos em ordem com ids sequenciais");
    assert!(grouped.iter().all(|chunk| chunk.paragraphs.clone().all(|at| document[at].page == chunk.page)), "chunker respeita limites de página");
    assert!(grouped.iter().all(|c| size(&c.text) <= 1600) && grouped.iter().any(|c| c.paragraphs.len() > 1) && grouped.len() < document.len(), "chunker agrupa até o alvo sem passar do máximo");
    assert!(grouped.iter().all(|chunk| chunk.paragraphs.clone().all(|at| chunk.text.contains(&document[at].original))), "chunker preserva o texto original dos parágrafos");
    assert!(rag::chunks(&[]).is_empty(), "chunker com documento vazio");
    let pages: Vec<usize> = rag::chunks(&[Paragraph::new(1, "A"), Paragraph::new(2, "B"), Paragraph::new(1, "C")]).iter().map(|c| c.page).collect();
    assert_eq!(pages, [1, 2, 1], "chunker não junta páginas iguais não consecutivas");

    let long = long_paragraph();
    let with_long = [Paragraph::new(1, "Short introduction."), Paragraph::new(1, long.clone()), Paragraph::new(1, "Short conclusion.")];
    let split = rag::chunks(&with_long);
    let pieces: Vec<&Chunk> = split.iter().filter(|c| c.paragraphs == (1..2)).collect();
    assert!(size(&long) > 1600 && pieces.len() >= 3 && pieces.iter().all(|c| size(&c.text) <= 1600), "parágrafo longo é dividido em vários chunks do mesmo parágrafo");
    assert!(pieces.iter().all(|c| c.text.ends_with('.') && c.text.starts_with("Sentence number")), "divisão acontece em fronteiras de frase");
    assert_eq!(pieces.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" "), long, "divisão não perde nem duplica texto");
    assert!(covered(&split, 3) && split.first().unwrap().text == "Short introduction." && split.last().unwrap().text == "Short conclusion.", "vizinhos do parágrafo longo continuam cobertos");
    let unbroken = "x".repeat(4000);
    let hard = rag::chunks(&[Paragraph::new(1, unbroken.clone())]);
    assert!(hard.len() == 3 && hard.iter().all(|c| size(&c.text) <= 1600 && c.paragraphs == (0..1)) && hard.iter().map(|c| c.text.as_str()).collect::<String>() == unbroken, "texto sem frases nem espaços ainda respeita o máximo");
    assert!(rag::chunks_with(&with_long, 200, 300).iter().all(|c| size(&c.text) <= 300), "limites personalizados são respeitados");

    let numeric = [
        Paragraph::new(1, "12"),
        Paragraph::new(1, "a".repeat(1000)),
        Paragraph::new(1, "3.450,00"),
        Paragraph::new(1, "b".repeat(1000)),
        Paragraph::new(1, "— 7 —"),
        Paragraph::new(2, "2025"),
        Paragraph::new(2, "• 14"),
        Paragraph::new(3, "99"),
        Paragraph::new(3, long.clone()),
        Paragraph::new(3, "100"),
    ];
    let chunks = rag::chunks(&numeric);
    assert!(covered(&chunks, numeric.len()), "parágrafos numéricos continuam cobertos");
    assert!(chunks.iter().filter(|c| c.page != 2).all(|c| c.text.chars().any(char::is_alphabetic)), "parágrafo sem letras nunca vira chunk isolado");
    let second: Vec<&Chunk> = chunks.iter().filter(|c| c.page == 2).collect();
    assert!(second.len() == 1 && second[0].paragraphs == (5..7) && second[0].text == "2025\n• 14", "página só com números gera um único chunk");
    assert!(
        chunks.first().unwrap().text.starts_with("12\n")
            && chunks.iter().any(|c| c.text.ends_with("\n— 7 —"))
            && chunks.iter().any(|c| c.text.starts_with("99\n"))
            && chunks.last().unwrap().text.ends_with("\n100"),
        "números se anexam ao vizinho anterior ou seguinte"
    );
    assert!(chunks.iter().all(|c| size(&c.text) <= 1600), "anexar números não ultrapassa o máximo");
}

fn chunk(id: usize, text: &str, page: usize) -> Chunk {
    Chunk { id, page, paragraphs: id..id + 1, text: text.to_string() }
}

fn search(index: &VectorIndex, vector: &[f32], text: &str) -> Vec<Retrieved> {
    index.search(vector, text, &HashMap::new(), rag::SEARCH_LIMIT, rag::MINIMUM_SCORE)
}

fn ids(results: &[Retrieved]) -> Vec<usize> {
    results.iter().map(|r| r.chunk.id).collect()
}

#[test]
fn indice_vetorial() {
    let corpus = vec![
        chunk(0, "Revenue grew twelve percent during the fiscal year.", 1),
        chunk(1, "A tradução automática preserva o sentido do texto.", 1),
        chunk(2, "Warranty covers manufacturing defects for two years.", 1),
        chunk(3, "Shipping takes five business days after payment.", 1),
    ];
    let none = HashMap::new();
    let index = VectorIndex::new(corpus.clone(), &[unit(&[1., 0., 0., 0.]), unit(&[0., 1., 0., 0.]), unit(&[0.6, 0., 0.8, 0.]), unit(&[0., 0., 0., 1.])]);
    let dense = index.search(&[2., 0., 0., 0.], "", &none, 6, 0.25);
    assert_eq!(ids(&dense), [0, 2], "ranking denso ordena por cosseno e descarta abaixo do mínimo");
    assert!((dense[0].score - 1.0).abs() < 1e-5 && (dense[1].score - 0.6).abs() < 1e-5, "score é o cosseno mesmo com consulta não normalizada");
    assert_eq!(ids(&index.search(&[1., 0., 0., 0.], "", &none, 6, 0.7)), [0], "filtro de score mínimo");
    assert!(index.search(&[1., 0., 0., 0.], "", &none, 1, 0.25).len() == 1 && index.search(&[1., 0., 0., 0.], "", &none, 0, 0.25).is_empty(), "limite de resultados");

    let lexical_only = search(&index, &[], "traducao");
    assert!(ids(&lexical_only) == [1] && lexical_only[0].score == 0.0, "BM25 ignora acentos e score é 0 sem vetor");
    assert_eq!(search(&index, &[], "TRADUÇÃO do texto").first().map(|r| r.chunk.id), Some(1), "BM25 ignora maiúsculas");
    assert!(search(&index, &[], "de a o que the of").is_empty() && search(&index, &[], "x y").is_empty(), "stopwords e tokens curtos não casam");
    assert_eq!(ids(&search(&index, &[1., 2.], "warranty")), [2], "vetor de dimensão errada cai para busca lexical");

    let rescued = search(&index, &[1., 0., 0., 0.], "shipping payment");
    assert!(ids(&rescued) == [0, 2, 3] && rescued[2].score == 0.0, "acerto só de palavra resgata o trecho abaixo do score mínimo, sem passar à frente do melhor trecho semântico");
    assert_eq!(ids(&search(&index, &[0.6, 0., 0.8, 0.], "revenue")), [0, 2], "RRF combina ranking denso e lexical");
    assert_eq!(ids(&search(&index, &[0.6, 0., 0.8, 0.], "nothing matches here")), [2, 0], "sem acerto lexical vale só o ranking denso");

    assert!(search(&index, &[], "garantia").is_empty(), "sem extraText a tradução não é encontrada");
    let extra: HashMap<usize, String> = [(2, "A garantia cobre defeitos de fabricação por dois anos.".to_string())].into_iter().collect();
    assert_eq!(ids(&index.search(&[], "garantia", &extra, 6, 0.25)), [2], "extraText entra no corpus lexical do chunk");
    assert_eq!(ids(&index.search(&[0., 1., 0., 0.], "fabricacao", &extra, 6, 0.25)), [1, 2], "extraText participa da fusão com o ranking denso");
    assert!(search(&VectorIndex::new(vec![], &[]), &[1.], "a").is_empty(), "índice vazio");
    assert_eq!(ids(&search(&VectorIndex::new(corpus, &[]), &[1., 0., 0., 0.], "warranty")), [2], "índice sem vetores ainda responde lexicalmente");

    // Regressão: muitos trechos com uma palavra da pergunta não podem expulsar o melhor trecho semântico.
    let mut texts: Vec<String> = (0..40).map(|n| format!("Unrelated passage number {n} about something else entirely.")).collect();
    for (row, text) in texts.iter_mut().enumerate().take(18).skip(10) {
        *text = format!("The total for item {row} is listed here, total again.");
    }
    texts[0] = "Sales grew strongly in the northern region last year.".into();
    let crowd: Vec<Chunk> = texts.iter().enumerate().map(|(id, text)| chunk(id, text, 1)).collect();
    let vectors: Vec<Vec<f32>> = (0..40)
        .map(|row| {
            let c: f32 = if row == 0 { 1.0 } else if (10..18).contains(&row) { 0.6 - row as f32 * 0.01 } else { 0.3 };
            vec![c, (1.0 - c * c).sqrt()]
        })
        .collect();
    let hits = VectorIndex::new(crowd, &vectors).search(&[1., 0.], "qual foi o total de vendas", &none, 6, 0.25);
    assert!(hits.len() == 6 && hits.iter().any(|h| h.chunk.id == 0), "o melhor trecho semântico continua entre os enviados ao modelo mesmo com oito acertos de palavra");
    assert!(hits.iter().any(|h| (10..18).contains(&h.chunk.id)), "acertos de palavra continuam participando do resultado");
}

#[test]
fn prompt_e_citacoes() {
    assert_eq!(rag::citations("A entrega leva cinco dias [2]. Veja também [1, 3] e [1][4].", 4), [1, 2, 3, 4], "citações simples, em lista e adjacentes");
    assert_eq!(rag::citations("Fora do intervalo [0], [7] e [2].", 3), [2], "citações fora do intervalo são ignoradas");
    assert_eq!(rag::citations("Texto [nota], [a1], [], [ 3 ] e [2;1].", 3), [1, 2, 3], "colchetes sem número não são citações");
    assert!(rag::citations("Resumo [1-3].", 5) == [1, 2, 3] && rag::citations("Sem fontes.", 3).is_empty() && rag::citations("[1]", 0).is_empty(), "intervalos, resposta sem citações e zero fontes");

    let sources = [Retrieved { chunk: chunk(3, "Shipping takes five business days.", 7), score: 0.8 }, Retrieved { chunk: chunk(0, "Revenue grew.", 2), score: 0.5 }];
    let history: Vec<OllamaMessage> = (1..=6).map(|n| OllamaMessage::new(if n % 2 == 1 { "user" } else { "assistant" }, format!("h{n}"))).collect();
    let prompt = rag::messages("Qual é o prazo?", &sources, &history);
    assert!(prompt.len() == 6 && prompt[0] == OllamaMessage::new("system", rag::SYSTEM_PROMPT) && prompt.last().unwrap().role == "user", "messages(): system, histórico e pergunta");
    assert_eq!(prompt[1..5].iter().map(|m| m.content.as_str()).collect::<Vec<_>>(), ["h3", "h4", "h5", "h6"], "messages(): só as 4 últimas mensagens do histórico");
    let content = &prompt.last().unwrap().content;
    assert!(content.contains("[1] (página 7)\nShipping takes five business days.") && content.contains("[2] (página 2)\nRevenue grew."), "messages(): trechos numerados com página");
    assert!(content.find("Qual é o prazo?").unwrap() > content.find("Revenue grew.").unwrap(), "messages(): pergunta vem depois dos trechos");
    assert!(rag::messages("Oi", &[], &[]).len() == 2 && !rag::SYSTEM_PROMPT.is_empty(), "messages(): sem histórico nem fontes");
}

#[test]
fn cliente_ollama_offline() {
    let installed: Vec<String> = vec!["llama3.2:3b".into(), "bge-m3:latest".into()];
    assert!(ollama::has_model("bge-m3", &installed) && ollama::has_model("bge-m3:latest", &installed) && ollama::has_model("llama3.2:3b", &installed), "hasModel: sem tag casa qualquer tag e com tag casa exato");
    assert!(
        ollama::has_model("llama3.2", &installed)
            && !ollama::has_model("llama3.2:1b", &installed)
            && !ollama::has_model("llama3", &installed)
            && !ollama::has_model("bge", &installed)
            && !ollama::has_model("", &installed)
            && !ollama::has_model("bge-m3", &[]),
        "hasModel: tag diferente, prefixo e vazio não casam"
    );
    assert!(OllamaError::ModelMissing("x".into()) != OllamaError::NotRunning, "OllamaError é comparável");
    assert!([OllamaError::NotRunning, OllamaError::ModelMissing("bge-m3".into()), OllamaError::Server("falha".into())].iter().all(|e| !e.to_string().is_empty()), "OllamaError tem mensagens");

    assert!(ollama::is_local_endpoint("http://127.0.0.1:11434") && ollama::is_local_endpoint("http://[::1]:11434") && ollama::is_local_endpoint("http://127.0.0.1"), "endpoint loopback é aceito");
    for remote in [
        "https://example.com",
        "http://example.com:11434",
        "https://127.0.0.1:11434",
        "http://127.0.0.1.example.com",
        "http://user@127.0.0.1:11434",
        "http://127.0.0.1:11434@example.com",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?x=1",
        "http://localhost.example.com",
        "http://[::1]example.com",
    ] {
        assert!(!ollama::is_local_endpoint(remote), "endpoint recusado: {remote}");
    }
    assert!(!ollama::is_local_model("llama:cloud") && ollama::is_local_model("llama3.2:3b") && !ollama::is_local_model("https://x/y"), "modelos cloud são recusados");

    let cancel = AtomicBool::new(false);
    let remote = OllamaClient::new("https://example.com");
    assert!(!remote.is_running(), "endpoint remoto não recebe requisição de disponibilidade");
    assert_eq!(remote.models(), Err(OllamaError::LocalOnly), "endpoint remoto não lista modelos");
    assert_eq!(remote.embed("bge-m3", &["a".into()], &cancel), Err(OllamaError::LocalOnly), "endpoint remoto não recebe texto do documento");
    let local = OllamaClient::default();
    assert_eq!(local.chat("llama:cloud", &[], ChatOptions::default(), &cancel, |_| {}), Err(OllamaError::LocalOnly), "modelo cloud não recebe conversa");

    let dead = OllamaClient::new("http://127.0.0.1:9");
    assert!(!dead.is_running(), "cliente em porta fechada: isRunning() é falso");
    assert_eq!(dead.embed("bge-m3", &["a".into()], &cancel), Err(OllamaError::NotRunning), "cliente em porta fechada: embed lança NotRunning");
    assert_eq!(dead.chat("llama3.2:3b", &[OllamaMessage::new("user", "oi")], ChatOptions::default(), &cancel, |_| {}), Err(OllamaError::NotRunning), "cliente em porta fechada: chat lança NotRunning");
    let cancelled = AtomicBool::new(true);
    assert_eq!(local.chat("llama3.2:3b", &[], ChatOptions::default(), &cancelled, |_| {}), Err(OllamaError::Cancelled), "pedido cancelado não chega ao servidor");
}

#[test]
fn intencao_da_pergunta() {
    for overview in ["Resuma o documento", "Faça uma síntese geral deste documento.", "Do que se trata?", "Quais são os pontos principais?", "Resumo", "Give me an overview of the file"] {
        assert!(assistant::wants_overview(overview), "pede síntese: {overview}");
    }
    for specific in ["Resuma a cláusula de garantia", "Resuma a política de garantia da empresa", "Qual é o prazo de entrega?", "Resuma a página 3", "Quem assina o contrato?"] {
        assert!(!assistant::wants_overview(specific), "não pede síntese: {specific}");
    }
    assert!(assistant::is_follow_up("E na Europa?") && assistant::is_follow_up("mas isso vale para todos os clientes do país?") && !assistant::is_follow_up("Quem assina o contrato pela empresa?"), "perguntas de continuação");

    let mut messages = vec![
        ChatMessage::new(1, Role::User, "Qual é o prazo?", Stage::Done, false),
        ChatMessage::new(2, Role::Assistant, "São dez dias [1, 2]. Veja [3].", Stage::Done, false),
        ChatMessage::new(3, Role::User, "Resuma o documento", Stage::Done, false),
        ChatMessage::new(4, Role::Assistant, "Síntese longa [1].", Stage::Done, true),
        ChatMessage::new(5, Role::Assistant, "falhou", Stage::Failed, false),
    ];
    let history = assistant::history(&messages);
    assert_eq!(history.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(), ["Qual é o prazo?", "São dez dias. Veja.", "Resuma o documento"], "histórico sem citações, sínteses nem falhas");
    assert_eq!(assistant::previous_question(&messages).as_deref(), Some("Qual é o prazo?"), "pergunta anterior ignora pedidos de síntese");
    messages[1].apply(assistant::Update::Finished { text: "Novo".into(), cited: vec![1], note: None });
    assert!(messages[1].text == "Novo" && messages[1].cited == [1] && messages[1].stage == Stage::Done, "atualização final é aplicada à mensagem");
    let chunks = [chunk(0, "a", 1), chunk(1, "b", 1)];
    assert!(assistant::index_key(&chunks).starts_with("bge-m3|chunker-1|2|") && assistant::index_key(&chunks) != assistant::index_key(&chunks[..1]), "chave do índice depende do texto");
}
