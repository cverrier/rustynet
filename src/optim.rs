//! The Adam optimizer — the blessed update rule.
//!
//! Adam keeps two running averages per parameter: the gradient (first moment `m`) and the squared
//! gradient (second moment `v`). Each step it bias-corrects them (they start at zero, so early
//! estimates are biased toward zero) and takes a step scaled by `m_hat / (sqrt(v_hat) + eps)`,
//! which adapts the effective learning rate per parameter. This is a faithful port of the update
//! in Karpathy's reference.

use crate::engine::Value;

/// Adam optimizer holding one first- and second-moment buffer per parameter.
pub struct Adam {
    params: Vec<Value>,
    /// First-moment (mean of gradients) buffer, one entry per parameter.
    m: Vec<f64>,
    /// Second-moment (mean of squared gradients) buffer, one entry per parameter.
    v: Vec<f64>,
    beta1: f64,
    beta2: f64,
    eps: f64,
    /// Number of steps taken so far, used for bias correction.
    t: u32,
}

impl Adam {
    /// Create an optimizer over `params` with the usual GPT-ish defaults
    /// (`beta1 = 0.85`, `beta2 = 0.99`, `eps = 1e-8`).
    pub fn new(params: Vec<Value>) -> Self {
        let n = params.len();
        Adam {
            params,
            m: vec![0.0; n],
            v: vec![0.0; n],
            beta1: 0.85,
            beta2: 0.99,
            eps: 1e-8,
            t: 0,
        }
    }

    /// Apply one Adam update using the gradients currently stored on the parameters.
    ///
    /// `lr` is passed per step so the caller can implement a learning-rate schedule (e.g. linear
    /// decay) without the optimizer needing to know about it.
    pub fn step(&mut self, lr: f64) {
        self.t += 1;
        // Bias-correction denominators for this step.
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, p) in self.params.iter().enumerate() {
            let g = p.grad();
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            p.set_data(p.data() - lr * m_hat / (v_hat.sqrt() + self.eps));
        }
    }

    /// Reset every parameter's gradient to zero, ready for the next forward/backward pass.
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimizing `f(x) = x^2` should drive `x` toward 0.
    #[test]
    fn descends_quadratic() {
        let x = Value::new(5.0);
        let mut opt = Adam::new(vec![x.clone()]);
        for _ in 0..250 {
            let loss = x.clone() * x.clone();
            loss.backward();
            opt.step(0.1);
            opt.zero_grad();
        }
        assert!(x.data().abs() < 1e-6, "x did not converge: {}", x.data());
    }
}
