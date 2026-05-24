//! Finite-difference gradient checks for the autograd engine.
//!
//! For each test we build an expression from leaf inputs, run [`Value::backward`] to get the
//! analytic gradients, then independently estimate each gradient numerically by nudging that input
//! by ±ε and measuring the change in the output: `(f(x+ε) - f(x-ε)) / 2ε`. The two should agree to
//! within a small tolerance — strong evidence that backprop is implemented correctly.

use rustynet::Value;
use rustynet::nn::{cross_entropy, softmax};

const EPS: f64 = 1e-6;
const TOL: f64 = 1e-4;

/// Compare analytic gradients (from `backward`) against central finite differences for `f`.
fn check<F>(f: F, inputs: &[f64])
where
    F: Fn(&[Value]) -> Value,
{
    // Analytic gradients via reverse-mode autodiff.
    let leaves: Vec<Value> = inputs.iter().map(|&x| Value::new(x)).collect();
    let out = f(&leaves);
    out.backward();
    let analytic: Vec<f64> = leaves.iter().map(Value::grad).collect();

    // Numerical gradients via central differences (graph rebuilt for each evaluation).
    for i in 0..inputs.len() {
        let eval = |delta: f64| {
            let mut perturbed = inputs.to_vec();
            perturbed[i] += delta;
            let leaves: Vec<Value> = perturbed.iter().map(|&x| Value::new(x)).collect();
            f(&leaves).data()
        };
        let numeric = (eval(EPS) - eval(-EPS)) / (2.0 * EPS);
        assert!(
            (numeric - analytic[i]).abs() < TOL,
            "input {i}: analytic {} vs numeric {}",
            analytic[i],
            numeric
        );
    }
}

#[test]
fn arithmetic_ops() {
    // f = a*b + a/b - b + 3*a
    check(
        |v| {
            v[0].clone() * v[1].clone() + v[0].clone() / v[1].clone() - v[1].clone()
                + 3.0 * v[0].clone()
        },
        &[1.5, 2.5],
    );
}

#[test]
fn powers_and_transcendentals() {
    // f = exp(a) + ln(b) + a^3
    check(|v| v[0].exp() + v[1].ln() + v[0].powf(3.0), &[0.4, 1.2]);
}

#[test]
fn relu_on_positive_branch() {
    // f = relu(a)*b ; keep `a` strictly positive so relu is differentiable here
    check(|v| v[0].relu() * v[1].clone(), &[0.8, -1.3]);
}

#[test]
fn softmax_cross_entropy() {
    // The real training objective: gradient of cross-entropy w.r.t. the logits.
    check(|logits| cross_entropy(logits, 2), &[0.5, -1.0, 2.0, 0.1]);
}

#[test]
fn softmax_output_component() {
    // Gradient of a single softmax probability w.r.t. the logits.
    check(|logits| softmax(logits)[1].clone(), &[1.0, 0.0, -0.5]);
}
