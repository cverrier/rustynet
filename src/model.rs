//! `NameMLP`: a Bengio-style character-level MLP language model — GPT without the attention.
//!
//! The model predicts the next character from a fixed window of the previous `block_size`
//! characters. Each context character is embedded, the embeddings are concatenated into one long
//! input vector, and a small MLP maps that to a distribution over the next character:
//!
//! ```text
//! [c_-3, c_-2, c_-1]  ->  embed each  ->  concat
//!                     ->  rmsnorm
//!                     ->  Linear (fc1) -> ReLU
//!                     ->  Linear (lm_head) -> logits over vocabulary
//! ```
//!
//! This is the historical predecessor of the transformer: it exercises embeddings, a linear layer,
//! normalization, a nonlinearity, and a softmax cross-entropy head — the core modern building
//! blocks — without any sequence-mixing attention.

use rand::Rng;

use crate::engine::Value;
use crate::nn::{Embedding, Linear, Module, relu, rmsnorm};

/// Standard deviation for Gaussian weight initialization (matches Karpathy's default).
const INIT_STD: f64 = 0.08;

/// A fixed-context character-level MLP language model.
pub struct NameMLP {
    embedding: Embedding,
    /// First MLP layer: maps the concatenated context embeddings up to the hidden width.
    fc1: Linear,
    /// Output head: maps the hidden activations to one logit per vocabulary token.
    lm_head: Linear,
    block_size: usize,
}

impl NameMLP {
    /// Build the model.
    ///
    /// - `vocab_size`: number of tokens (including BOS).
    /// - `block_size`: how many previous characters the model conditions on.
    /// - `n_embd`: embedding width per character.
    /// - `hidden`: width of the MLP hidden layer.
    pub fn new(
        vocab_size: usize,
        block_size: usize,
        n_embd: usize,
        hidden: usize,
        rng: &mut impl Rng,
    ) -> Self {
        NameMLP {
            embedding: Embedding::new(vocab_size, n_embd, INIT_STD, rng),
            fc1: Linear::new(hidden, block_size * n_embd, INIT_STD, rng),
            lm_head: Linear::new(vocab_size, hidden, INIT_STD, rng),
            block_size,
        }
    }

    /// The context length the model expects.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Forward a single context window (length `block_size`) into next-token logits.
    pub fn forward(&self, context: &[usize]) -> Vec<Value> {
        debug_assert_eq!(
            context.len(),
            self.block_size,
            "context must be block_size long"
        );

        // Embed each context token and concatenate into one input vector.
        let mut x: Vec<Value> = Vec::with_capacity(self.block_size);
        for &token in context {
            x.extend(self.embedding.forward(token));
        }

        let x = rmsnorm(&x);
        let h = relu(&self.fc1.forward(&x));
        self.lm_head.forward(&h)
    }
}

impl Module for NameMLP {
    fn parameters(&self) -> Vec<Value> {
        let mut params = self.embedding.parameters();
        params.extend(self.fc1.parameters());
        params.extend(self.lm_head.parameters());
        params
    }
}
