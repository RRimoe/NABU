//! Verifica che il catalogo dei modelli sia allineato alla realtà.
//!
//! Richiede rete, quindi è `#[ignore]` per default:
//!     cargo test --test registry_urls -- --ignored --nocapture

use sha2::{Digest, Sha256};

#[path = "../src/ai/registry.rs"]
mod registry;

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("Nabu-AI-Editor")
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .expect("client HTTP")
}

/// Gli id devono essere unici: sono usati come nome di cartella su disco.
#[test]
fn model_ids_are_unique() {
    let mut ids: Vec<&str> = registry::MODELS.iter().map(|m| m.id).collect();
    let total = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), total, "id duplicati nel catalogo");
}

/// Ogni modello deve avere almeno un file e nomi di file distinti.
#[test]
fn model_files_are_well_formed() {
    for spec in registry::MODELS {
        assert!(!spec.files.is_empty(), "{} non ha file", spec.id);

        let mut names: Vec<&str> = spec.files.iter().map(|f| f.file_name).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "{} ha nomi di file duplicati", spec.id);

        for file in spec.files {
            assert!(
                file.url.starts_with("https://"),
                "{} usa un URL non cifrato",
                spec.id
            );
            assert!(file.size_bytes > 0, "{} dichiara dimensione zero", spec.id);
            if let Some(hash) = file.sha256 {
                assert_eq!(hash.len(), 64, "{}: SHA-256 malformato", spec.id);
                assert!(
                    hash.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
                    "{}: SHA-256 non in esadecimale minuscolo",
                    spec.id
                );
            }
        }
    }
}

/// Ogni URL deve rispondere e dichiarare la dimensione attesa dal catalogo.
#[tokio::test]
#[ignore = "richiede accesso di rete"]
async fn every_url_is_reachable_with_expected_size() {
    let client = client();

    for spec in registry::MODELS {
        for file in spec.files {
            let response = client
                .head(file.url)
                .send()
                .await
                .unwrap_or_else(|e| panic!("{} / {}: {e}", spec.id, file.file_name));

            assert!(
                response.status().is_success(),
                "{} / {}: risposta {}",
                spec.id,
                file.file_name,
                response.status()
            );

            let len: u64 = response
                .headers()
                .get(reqwest::header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);

            assert_eq!(
                len, file.size_bytes,
                "{} / {}: dimensione remota diversa da quella nel catalogo",
                spec.id, file.file_name
            );

            println!("ok  {:<26} {:>12} B  {}", spec.id, len, file.file_name);
        }
    }
}

/// Scarica il modello più piccolo e ne verifica il digest: convalida in un
/// colpo solo la configurazione TLS, il redirect di Hugging Face e l'hash.
#[tokio::test]
#[ignore = "richiede accesso di rete"]
async fn smallest_model_matches_its_digest() {
    let spec = registry::MODELS
        .iter()
        .min_by_key(|m| m.total_bytes())
        .expect("catalogo non vuoto");
    let file = &spec.files[0];
    let expected = file.sha256.expect("il modello di test deve avere un digest");

    println!("scarico {} ({} B)...", spec.id, file.size_bytes);
    let bytes = client()
        .get(file.url)
        .send()
        .await
        .expect("richiesta")
        .error_for_status()
        .expect("stato HTTP")
        .bytes()
        .await
        .expect("corpo della risposta");

    assert_eq!(bytes.len() as u64, file.size_bytes, "dimensione inattesa");

    let actual = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(actual, expected, "digest non corrispondente per {}", spec.id);
    println!("digest verificato: {actual}");
}
