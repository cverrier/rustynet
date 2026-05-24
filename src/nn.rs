//! Neural-network building blocks: trainable layers, activations, normalization, and the loss.
//!
//! Everything operates on vectors of [`Value`]s (`Vec<Value>` / `&[Value]`) — there is no tensor
//! abstraction on purpose. Each scalar carries its own slice of the computation graph, so the
//! autograd engine handles all the gradients for free.

use rand::Rng;
use rand::distr::Distribution;
use rand_distr::Normal;

use crate::engine::Value;

/// Anything that owns trainable parameters. The optimizer asks a model for this flat list of
/// [`Value`]s; because clones share the same node, updating them updates the live model.
pub trait Module {
    /// Every trainable scalar in the module, in a stable order.
    fn parameters(&self) -> Vec<Value>;
}

/// A bias-free linear (matrix–vector) layer: `y = W x`, with `W` of shape `n_out × n_in`.
///
/// Weights are initialized from a small Gaussian, matching Karpathy's `random.gauss(0, std)`.
pub struct Linear {
    /// Row-major weights: `w[o][i]` multiplies input `i` to contribute to output `o`.
    w: Vec<Vec<Value>>,
}

impl Linear {
    /// Create a layer with Gaussian-initialized weights (standard deviation `std`).
    pub fn new(n_out: usize, n_in: usize, std: f64, rng: &mut impl Rng) -> Self {
        let normal = Normal::new(0.0, std).expect("std must be finite and non-negative");
        let w = (0..n_out)
            .map(|_| (0..n_in).map(|_| Value::new(normal.sample(rng))).collect())
            .collect();
        Linear { w }
    }

    /// Forward pass: each output is the dot product of a weight row with the input vector.
    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.w
            .iter()
            .map(|row| {
                row.iter()
                    .zip(x.iter())
                    .fold(Value::new(0.0), |acc, (wi, xi)| {
                        acc + wi.clone() * xi.clone()
                    })
            })
            .collect()
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Value> {
        self.w.iter().flatten().cloned().collect()
    }
}

/// A lookup table mapping a token id to a learned `dim`-dimensional embedding vector.
pub struct Embedding {
    /// `table[id]` is the embedding row for token `id`.
    table: Vec<Vec<Value>>,
}

impl Embedding {
    /// Create a `vocab_size × dim` embedding table, Gaussian-initialized.
    pub fn new(vocab_size: usize, dim: usize, std: f64, rng: &mut impl Rng) -> Self {
        let normal = Normal::new(0.0, std).expect("std must be finite and non-negative");
        let table = (0..vocab_size)
            .map(|_| (0..dim).map(|_| Value::new(normal.sample(rng))).collect())
            .collect();
        Embedding { table }
    }

    /// Look up the embedding for `id`. The returned [`Value`]s are clones, so they share — and
    /// receive gradients into — the underlying table entries.
    pub fn forward(&self, id: usize) -> Vec<Value> {
        self.table[id].clone()
    }
}

impl Module for Embedding {
    fn parameters(&self) -> Vec<Value> {
        self.table.iter().flatten().cloned().collect()
    }
}

/// Apply ReLU element-wise.
pub fn relu(x: &[Value]) -> Vec<Value> {
    x.iter().map(Value::relu).collect()
}

/// Root-mean-square normalization (the modern, bias/mean-free cousin of LayerNorm).
///
/// Scales the vector so its root-mean-square is ~1: `x_i * (mean(x^2) + eps)^(-1/2)`.
pub fn rmsnorm(x: &[Value]) -> Vec<Value> {
    // TODO: Avoid using `.fold()` if possible, so it looks nicer
    let n = x.len() as f64;
    let sum_sq = x
        .iter()
        .fold(Value::new(0.0), |acc, xi| acc + xi.clone() * xi.clone());
    let ms = sum_sq * (1.0 / n);
    // (mean_square + eps) ** -0.5, computed through the graph so it is differentiable.
    let scale = (ms + 1e-5).powf(-0.5);
    x.iter().map(|xi| xi.clone() * scale.clone()).collect()
}

/// Numerically stable softmax: subtract the max logit before exponentiating.
pub fn softmax(logits: &[Value]) -> Vec<Value> {
    let max_val = logits
        .iter()
        .map(Value::data)
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<Value> = logits.iter().map(|v| (v.clone() - max_val).exp()).collect();
    let total = exps.iter().fold(Value::new(0.0), |acc, e| acc + e.clone());
    exps.into_iter().map(|e| e / total.clone()).collect()
}

/// Cross-entropy loss for a single example: the negative log-probability the model assigns to the
/// correct class. `target` indexes into `logits`.
pub fn cross_entropy(logits: &[Value], target: usize) -> Value {
    // TODO: Do not materialize the full `probs` array if possible
    let probs = softmax(logits);
    -probs[target].clone().ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![Value::new(1.0), Value::new(2.0), Value::new(3.0)];
        let probs = softmax(&logits);
        let total: f64 = probs.iter().map(Value::data).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn linear_shapes_and_grad_flow() {
        let mut rng = StdRng::seed_from_u64(0);
        let layer = Linear::new(2, 3, 0.1, &mut rng);
        let x = vec![Value::new(1.0), Value::new(2.0), Value::new(3.0)];
        let y = layer.forward(&x);
        assert_eq!(y.len(), 2);
        // A scalar loss should produce non-trivial gradients on the weights.
        let loss = y.into_iter().fold(Value::new(0.0), |a, b| a + b);
        loss.backward();
        assert_eq!(layer.parameters().len(), 6);
    }

    #[test]
    fn cross_entropy_low_when_confident_correct() {
        let logits = vec![Value::new(10.0), Value::new(0.0), Value::new(0.0)];
        let loss = cross_entropy(&logits, 0);
        assert!(loss.data() < 0.01);
    }
}
