//! Verificações ao vivo com o Ollama real (porte de `Checks/RAG` e `Checks/Assistant`).
//! Pulam quando o servidor não está em execução ou os modelos não estão instalados.

use paragrafo_core::assistant::{self, Availability, Overview, Question, Stage, Update};
use paragrafo_core::extraction::Paragraph;
use paragrafo_core::library::{Library, SavedDocument};
use paragrafo_core::ollama::{self, ChatOptions, OllamaClient, OllamaError, OllamaMessage};
use paragrafo_core::rag::{self, VectorIndex};
use paragrafo_core::translate;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const CHAT: &str = "llama3.2:3b";
const EMBED: &str = "bge-m3";

fn ready() -> Option<OllamaClient> {
    let client = OllamaClient::default();
    if !client.is_running() {
        println!("SKIP: Ollama não está em execução; verificações ao vivo ignoradas (inicie com `ollama serve`)");
        return None;
    }
    let available = client.models().expect("models() lista os modelos instalados");
    if !ollama::has_model(EMBED, &available) || !ollama::has_model(CHAT, &available) {
        println!("SKIP: modelos {EMBED} e {CHAT} não instalados; verificações ao vivo ignoradas");
        return None;
    }
    Some(client)
}

fn topics() -> Vec<Paragraph> {
    [
        "Total revenue for the fiscal year reached 4.2 million dollars, an increase of twelve percent over the previous year.",
        "Our privacy policy states that personal data is never sold to third parties and is stored encrypted on servers in the European Union.",
        "Orders are shipped within two business days and delivery takes at most ten business days after payment is confirmed.",
        "The product warranty covers manufacturing defects for twenty-four months from the date of purchase.",
        "Employees are entitled to thirty days of paid vacation per year, which must be scheduled with their manager.",
        "The software requires at least 8 GB of memory and runs on a recent Linux distribution.",
        "Customer support is available by email from Monday to Friday, between nine in the morning and six in the evening.",
        "The board of directors meets quarterly to review strategy and approve the annual budget.",
    ]
    .iter()
    .enumerate()
    .map(|(offset, text)| Paragraph::new(offset + 1, *text))
    .collect()
}

#[test]
fn cliente_ao_vivo() {
    let Some(client) = ready() else { return };
    let go = AtomicBool::new(false);
    let three = client.embed(EMBED, &["Hello world.".into(), "Olá, mundo.".into(), "The warranty lasts two years.".into()], &go).unwrap();
    let norm = |v: &Vec<f32>| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(three.len() == 3 && !three[0].is_empty() && three.iter().all(|v| v.len() == three[0].len() && (norm(v) - 1.0).abs() < 1e-3), "embed devolve 3 vetores normalizados de mesma dimensão");
    assert!(client.embed(EMBED, &[], &go).unwrap().is_empty(), "embed sem entradas devolve vazio");

    let chunks = rag::chunks(&topics());
    assert_eq!(chunks.len(), 8, "um chunk por página no documento de teste");
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vectors = client.embed(EMBED, &texts, &go).unwrap();
    let question = "Qual é o prazo de entrega?";
    let query = client.embed(EMBED, &[question.to_string()], &go).unwrap().remove(0);
    let index = VectorIndex::new(chunks, &vectors);
    let hits = index.search(&query, question, &HashMap::new(), rag::SEARCH_LIMIT, rag::MINIMUM_SCORE);
    assert!(hits.first().is_some_and(|h| h.chunk.text.contains("shipped")), "recuperação multilíngue: pergunta em português encontra o trecho de entrega em inglês");
    assert!(hits[0].score > 0.4, "cosseno do melhor trecho acima de 0,4");

    let mut answer = String::new();
    let mut deltas = 0;
    let options = ChatOptions { temperature: 0.0, ..ChatOptions::default() };
    client.chat(CHAT, &rag::messages(question, &hits, &[]), options, &go, |delta| {
        deltas += 1;
        answer.push_str(delta);
    })
    .unwrap();
    println!("INFO: resposta: {}", answer.replace('\n', " "));
    assert!(!answer.trim().is_empty() && deltas > 1, "chat em streaming produz resposta em mais de um delta");
    let cited = rag::citations(&answer, hits.len());
    assert!(cited.contains(&1) && (answer.contains("dez") || answer.contains("10")), "resposta usa o trecho de entrega e responde em português");

    // Cancelar no meio do stream encerra a leitura e o servidor continua responsivo.
    let stop = AtomicBool::new(false);
    let mut received = 0;
    let started = Instant::now();
    let essay = [OllamaMessage::new("user", "Escreva um texto de 800 palavras sobre a história da navegação.")];
    let result = client.chat(CHAT, &essay, ChatOptions::default(), &stop, |_| {
        received += 1;
        if received == 5 {
            stop.store(true, Ordering::Relaxed);
        }
    });
    assert!(result == Err(OllamaError::Cancelled) && (5..200).contains(&received), "cancelar encerra o stream de chat ({received} deltas)");
    println!("INFO: cancelamento no meio do chat retornou em {:.3} s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    client.embed(EMBED, &["ping".into()], &go).unwrap();
    assert!(started.elapsed().as_secs() < 10, "servidor continua responsivo depois do cancelamento");

    assert_eq!(client.embed("modelo-inexistente-xyz", &["a".into()], &go), Err(OllamaError::ModelMissing("modelo-inexistente-xyz".into())), "embed com modelo inexistente");
    assert_eq!(client.chat("modelo-inexistente-xyz", &[OllamaMessage::new("user", "oi")], ChatOptions::default(), &go, |_| {}), Err(OllamaError::ModelMissing("modelo-inexistente-xyz".into())), "chat com modelo inexistente");
    let mut progress: Vec<f64> = Vec::new();
    client.pull(EMBED, &go, |value| progress.push(value)).unwrap();
    assert!(progress.last() == Some(&1.0) && progress.windows(2).all(|p| p[0] <= p[1]) && progress.iter().all(|v| (0.0..=1.0).contains(v)), "pull de modelo já instalado termina em 1 com progresso crescente");
}

#[test]
fn assistente_ao_vivo() {
    let Some(client) = ready() else { return };
    let root = std::env::temp_dir().join(format!("paragrafo-live-{}", std::process::id()));
    let library = Library::at(&root);
    let go = AtomicBool::new(false);
    let paragraphs = topics();
    let chunks = rag::chunks(&paragraphs);
    let fingerprint = "live-check";
    library.save(&SavedDocument::new("teste.pdf", "/tmp/teste.pdf", 8, "en", "pt", vec![String::new(); paragraphs.len()]), fingerprint);

    let checked = assistant::check(&client, "modelo-que-nao-existe", false);
    assert!(checked.availability == Availability::Ready && checked.chat_model != "modelo-que-nao-existe" && checked.chat_models.iter().all(|m| !m.contains("bge")), "disponibilidade: pronto, e a escolha inválida cai para um modelo instalado");

    let mut steps = Vec::new();
    let index = assistant::build_index(&client, &library, fingerprint, &chunks, &go, |done, total| steps.push((done, total))).unwrap();
    assert!(steps.first() == Some(&(0, 8)) && steps.last() == Some(&(8, 8)), "indexação relata o progresso");
    let saved = library.load(fingerprint).unwrap();
    assert!(saved.index_key.as_deref() == Some(assistant::index_key(&chunks).as_str()) && saved.dimension.is_some(), "índice fica salvo com a chave do texto");
    let mut again = Vec::new();
    assistant::build_index(&client, &library, fingerprint, &chunks, &go, |done, total| again.push((done, total))).unwrap();
    assert!(again.is_empty(), "reabrir restaura o índice sem recalcular");

    let mut updates = Vec::new();
    let question = Question { text: "Por quanto tempo vale a garantia?", previous: None, history: &[], translations: &[] };
    assistant::answer(&client, &index, CHAT, &question, &go, &mut |update| updates.push(update)).unwrap();
    let Some(Update::Finished { text, cited, .. }) = updates.last().cloned() else { panic!("resposta não terminou") };
    println!("INFO: resposta do assistente: {}", text.replace('\n', " "));
    assert!(updates.iter().any(|u| matches!(u, Update::Sources(s) if s[0].chunk.text.contains("warranty"))), "o trecho da garantia é o primeiro consultado");
    assert!(updates.contains(&Update::Stage(Stage::Writing)) && !cited.is_empty() && (text.contains("24") || text.to_lowercase().contains("vinte")), "resposta cita a fonte e traz o prazo");

    // Síntese em uma passada, salva e depois restaurada sem chamar o modelo.
    let request = Overview { fingerprint, paragraphs: &paragraphs, chunks: &chunks, chat_model: CHAT, question: assistant::OVERVIEW_QUESTION, force: false, single_pass_limit: assistant::SINGLE_PASS_LIMIT };
    let mut first = Vec::new();
    assistant::overview(&client, &library, &request, &go, &mut |update| first.push(update)).unwrap();
    let Some(Update::Finished { text: summary, note, .. }) = first.last().cloned() else { panic!("síntese não terminou") };
    assert!(!summary.is_empty() && note.is_none() && library.load(fingerprint).unwrap().summary.as_deref() == Some(summary.as_str()), "síntese é escrita e salva");
    let mut restored = Vec::new();
    assistant::overview(&client, &library, &request, &go, &mut |update| restored.push(update)).unwrap();
    assert!(matches!(restored.last(), Some(Update::Finished { text, note: Some(note), .. }) if *text == summary && note.contains("salva")), "síntese salva volta na hora");

    // Documento "longo": lido em partes e depois combinado.
    let parts = Overview { question: "Resuma o documento.", single_pass_limit: 400, ..request };
    let mut staged = Vec::new();
    assistant::overview(&client, &library, &parts, &go, &mut |update| staged.push(update)).unwrap();
    assert!(staged.iter().any(|u| matches!(u, Update::Stage(Stage::Reading(_, total)) if *total > 1)) && matches!(staged.last(), Some(Update::Finished { text, .. }) if !text.is_empty()), "documento longo é lido em partes");

    let stopped = AtomicBool::new(true);
    assert_eq!(assistant::answer(&client, &index, CHAT, &question, &stopped, &mut |_| {}), Err(OllamaError::Cancelled), "parar interrompe a resposta");

    let translation = translate::translate(&client, CHAT, "en", "pt", "The warranty lasts two years.", &go).unwrap();
    assert!(!translation.kept_original, "o modelo respondeu com uma tradução");
    let translated = translation.text;
    println!("INFO: tradução: {translated}");
    assert!(translated.to_lowercase().contains("garantia") && translated.to_lowercase().contains("anos"), "tradução local para o português");
    std::fs::remove_dir_all(root).unwrap();
}
