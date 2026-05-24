//! Demo: train the [`NameMLP`] character-level language model on a list of names, then let it
//! babble brand-new, hallucinated names.
//!
//! This wires together every building block in the crate — tokenizer, embeddings, linear layers,
//! RMSNorm, softmax cross-entropy, autograd, and Adam — into one runnable program. Run it with:
//!
//! ```text
//! cargo run --release
//! ```

use std::process;

use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use rustynet::Value;
use rustynet::model::NameMLP;
use rustynet::nn::{Module, cross_entropy, softmax};
use rustynet::optim::Adam;
use rustynet::tokenizer::Tokenizer;

// --- Hyperparameters (kept small: the scalar autograd is built for clarity, not speed) ---------
const BLOCK_SIZE: usize = 3; // characters of context the model conditions on
const N_EMBD: usize = 16; // embedding width per character
const HIDDEN: usize = 64; // MLP hidden-layer width
const NUM_STEPS: usize = 1000; // training steps (one name per step)
const LEARNING_RATE: f64 = 0.01;
const NUM_SAMPLES: usize = 20; // names to hallucinate at the end
const TEMPERATURE: f64 = 0.5; // sampling "creativity" in (0, 1]
const DATA_PATH: &str = "data/names.txt";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(42); // let there be order among chaos

    // --- Dataset --------------------------------------------------------------------------------
    let mut docs = load_docs(DATA_PATH)?;
    docs.shuffle(&mut rng);
    println!("num docs: {}", docs.len());

    let tokenizer = Tokenizer::from_docs(&docs);
    let vocab_size = tokenizer.vocab_size();
    println!("vocab size: {vocab_size}");

    // --- Model & optimizer ----------------------------------------------------------------------
    let model = NameMLP::new(vocab_size, BLOCK_SIZE, N_EMBD, HIDDEN, &mut rng);
    let mut optimizer = Adam::new(model.parameters());
    println!("num params: {}", model.parameters().len());

    // --- Training -------------------------------------------------------------------------------
    println!("--- training ---");
    for step in 0..NUM_STEPS {
        let doc = &docs[step % docs.len()];
        let examples = make_examples(doc, &tokenizer);

        // Forward every (context, target) example, building one big graph up to the average loss.
        let losses: Vec<Value> = examples
            .iter()
            .map(|(context, target)| cross_entropy(&model.forward(context), *target))
            .collect();
        let n = losses.len() as f64;
        let loss = losses.into_iter().fold(Value::new(0.0), |a, b| a + b) * (1.0 / n);

        // Backward, then take an Adam step with a linearly decaying learning rate.
        loss.backward();
        let lr_t = LEARNING_RATE * (1.0 - step as f64 / NUM_STEPS as f64);
        optimizer.step(lr_t);
        optimizer.zero_grad();

        print!(
            "\rstep {:4} / {:4} | loss {:.4}",
            step + 1,
            NUM_STEPS,
            loss.data()
        );
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!();

    // --- Inference ------------------------------------------------------------------------------
    println!("--- inference (new, hallucinated names) ---");
    for i in 0..NUM_SAMPLES {
        let name = sample_name(&model, &tokenizer, TEMPERATURE, &mut rng);
        println!("sample {:2}: {name}", i + 1);
    }

    Ok(())
}

/// Read the dataset: one document (name) per non-empty line.
fn load_docs(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        format!(
            "could not read {path}: {e}\n\
             fetch the dataset first, e.g.:\n  \
             mkdir -p data && curl -L -o {path} \
             https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        )
    })?;
    let docs: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect();
    if docs.is_empty() {
        return Err(format!("{path} contained no usable lines"));
    }
    Ok(docs)
}

/// Turn one document into `(context, target)` training pairs via a sliding window.
///
/// The context starts as all-BOS and slides forward one character at a time; the model learns to
/// predict each character, and finally the terminating BOS, from the previous `BLOCK_SIZE` tokens.
fn make_examples(doc: &str, tokenizer: &Tokenizer) -> Vec<(Vec<usize>, usize)> {
    let bos = tokenizer.bos();
    let mut context = vec![bos; BLOCK_SIZE];
    let mut examples = Vec::new();

    let targets = doc
        .chars()
        .map(|ch| tokenizer.encode(ch))
        .chain(std::iter::once(bos)); // predict end-of-name too
    for target in targets {
        examples.push((context.clone(), target));
        context.remove(0);
        context.push(target);
    }
    examples
}

/// Autoregressively sample one name: feed the model its own predictions until it emits BOS.
fn sample_name(
    model: &NameMLP,
    tokenizer: &Tokenizer,
    temperature: f64,
    rng: &mut impl Rng,
) -> String {
    let bos = tokenizer.bos();
    let mut context = vec![bos; model.block_size()];
    let mut name = String::new();

    // Cap the length so an unlucky model can't loop forever.
    for _ in 0..32 {
        let logits = model.forward(&context);
        // Apply temperature, then softmax to get a probability distribution.
        let scaled: Vec<Value> = logits
            .iter()
            .map(|l| l.clone() * (1.0 / temperature))
            .collect();
        let probs: Vec<f64> = softmax(&scaled).iter().map(Value::data).collect();

        let next = WeightedIndex::new(&probs)
            .expect("probabilities are valid weights")
            .sample(rng);
        if next == bos {
            break;
        }
        if let Some(ch) = tokenizer.decode(next) {
            name.push(ch);
        }
        context.remove(0);
        context.push(next);
    }
    name
}
