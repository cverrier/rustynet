//! # rustynet
//!
//! A from-scratch, dependency-light implementation of the **core building blocks of a modern
//! neural network**, written for learning rather than speed.
//!
//! It is inspired by Andrej Karpathy's "most atomic" pure-Python GPT, but deliberately leaves out
//! the transformer machinery (attention, KV-cache, positional embeddings as used in transformers).
//! What remains is the part that is *the algorithm*: a scalar reverse-mode autograd engine, a few
//! neural-network layers, normalization, a softmax cross-entropy loss, and the Adam optimizer.
//!
//! The pieces are wired together in the binary (`main.rs`) into a Bengio-style character-level MLP
//! language model that learns to babble new names — the historical predecessor of GPT, and the
//! most natural illustration of these building blocks without any attention.
//!
//! Module map:
//! - [`engine`]: the autograd engine ([`engine::Value`]) — the heart of the project.
//! - [`nn`]: layers ([`nn::Linear`], [`nn::Embedding`]), activations, normalization, and the loss.
//! - [`optim`]: the [`optim::Adam`] optimizer.
//! - [`tokenizer`]: character ↔ integer-token translation.
//! - [`model`]: the concrete [`model::NameMLP`] demonstration network.

pub mod engine;
pub mod model;
pub mod nn;
pub mod optim;
pub mod tokenizer;

pub use engine::Value;
