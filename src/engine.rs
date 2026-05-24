//! The autograd engine — this part is *the algorithm*.
//!
//! A [`Value`] is one scalar node in a computation graph. Every arithmetic operation builds a new
//! node that remembers its children (the inputs) together with the *local derivative* of the
//! result with respect to each child, evaluated during the forward pass. This is the one elegant
//! trick from Karpathy's implementation: because the local gradients are stored at forward time,
//! the backward pass is a single uniform rule applied to every node —
//! `child.grad += local_grad * node.grad` — i.e. the chain rule, accumulated over the graph.
//!
//! Sharing and mutation are needed because a graph is a DAG (one node can feed many others) whose
//! gradients are filled in later, so each node is an `Rc<RefCell<…>>`. Cloning a [`Value`] is cheap
//! and shares the same underlying node — that is how the optimizer keeps a handle on every
//! parameter.

use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// A scalar node in the computation graph.
///
/// Clone is cheap (it bumps a reference count) and clones share the same underlying node, so a
/// gradient written through one clone is visible through all of them.
#[derive(Clone)]
pub struct Value(Rc<RefCell<Inner>>);

struct Inner {
    /// Scalar value computed during the forward pass.
    data: f64,
    /// Derivative of the final loss with respect to this node, filled in by [`Value::backward`].
    grad: f64,
    /// Inputs to the operation that produced this node.
    children: Vec<Value>,
    /// Local derivative of this node with respect to each child, computed at forward time.
    local_grads: Vec<f64>,
}

impl Value {
    /// Create a leaf node (a parameter or an input) with no children.
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(Inner {
            data,
            grad: 0.0,
            children: Vec::new(),
            local_grads: Vec::new(),
        })))
    }

    /// Build the node produced by an operation, recording its children and local gradients.
    fn from_op(data: f64, children: Vec<Value>, local_grads: Vec<f64>) -> Self {
        Value(Rc::new(RefCell::new(Inner {
            data,
            grad: 0.0,
            children,
            local_grads,
        })))
    }

    /// The current scalar value.
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    /// The gradient accumulated by the most recent [`backward`](Value::backward) call.
    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    /// Overwrite the scalar value (used by the optimizer when applying an update).
    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    /// Reset the gradient to zero (used by the optimizer between steps).
    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    /// Raise to a constant power: `self ** n`. Local grad: `n * self^(n-1)`.
    pub fn powf(&self, n: f64) -> Value {
        let x = self.data();
        Value::from_op(x.powf(n), vec![self.clone()], vec![n * x.powf(n - 1.0)])
    }

    /// Natural exponential. Local grad of `e^x` is `e^x`.
    pub fn exp(&self) -> Value {
        let e = self.data().exp();
        Value::from_op(e, vec![self.clone()], vec![e])
    }

    /// Natural logarithm. Local grad of `ln(x)` is `1/x`.
    pub fn ln(&self) -> Value {
        let x = self.data();
        Value::from_op(x.ln(), vec![self.clone()], vec![1.0 / x])
    }

    /// Rectified linear unit. Local grad is 1 where the input is positive, else 0.
    pub fn relu(&self) -> Value {
        let x = self.data();
        Value::from_op(
            x.max(0.0),
            vec![self.clone()],
            vec![if x > 0.0 { 1.0 } else { 0.0 }],
        )
    }

    /// Run reverse-mode automatic differentiation, filling in [`grad`](Value::grad) for every node
    /// reachable from `self`. Treats `self` as the output (its gradient seeds at 1).
    ///
    /// Assumes gradients start at zero, which holds because every forward pass builds fresh nodes.
    pub fn backward(&self) {
        // Build a topological order so that every node is visited only after all of its
        // descendants. We track which nodes we have already queued by raw pointer identity — the
        // direct Rust analogue of Python's `set()` of object ids.
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: HashSet<*const RefCell<Inner>> = HashSet::new();

        fn build_topo(
            v: &Value,
            topo: &mut Vec<Value>,
            visited: &mut HashSet<*const RefCell<Inner>>,
        ) {
            if visited.insert(Rc::as_ptr(&v.0)) {
                for child in v.0.borrow().children.iter() {
                    build_topo(child, topo, visited);
                }
                topo.push(v.clone());
            }
        }
        build_topo(self, &mut topo, &mut visited);

        self.0.borrow_mut().grad = 1.0;
        // Walk in reverse topological order, pushing each node's gradient onto its children.
        for v in topo.iter().rev() {
            let node = v.0.borrow();
            let upstream = node.grad;
            for (child, &local_grad) in node.children.iter().zip(node.local_grads.iter()) {
                // `child` is always a distinct node from `v` (the graph is acyclic), so borrowing
                // it mutably while `v` is borrowed immutably never aliases.
                child.0.borrow_mut().grad += local_grad * upstream;
            }
        }
    }
}

// --- Operator overloads -----------------------------------------------------------------------
//
// Each binary op records both children and their local gradients. We implement the owned `Value`
// combinations plus the `Value`/`f64` mixes so model code reads like ordinary arithmetic. `Value`
// is cheap to clone, so reusing an operand (e.g. a residual) is just an explicit `.clone()`.

impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        Value::from_op(
            self.data() + other.data(),
            vec![self, other],
            vec![1.0, 1.0],
        )
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        let (a, b) = (self.data(), other.data());
        // d(a*b)/da = b, d(a*b)/db = a
        Value::from_op(a * b, vec![self, other], vec![b, a])
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        self * Value::new(-1.0)
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        self + (-other)
    }
}

impl Div for Value {
    type Output = Value;
    fn div(self, other: Value) -> Value {
        self * other.powf(-1.0)
    }
}

// Scalar conveniences: `Value op f64` and `f64 op Value`.
macro_rules! impl_scalar_ops {
    ($trait:ident, $method:ident) => {
        impl $trait<f64> for Value {
            type Output = Value;
            fn $method(self, other: f64) -> Value {
                self.$method(Value::new(other))
            }
        }
        impl $trait<Value> for f64 {
            type Output = Value;
            fn $method(self, other: Value) -> Value {
                Value::new(self).$method(other)
            }
        }
    };
}
impl_scalar_ops!(Add, add);
impl_scalar_ops!(Mul, mul);
impl_scalar_ops!(Sub, sub);
impl_scalar_ops!(Div, div);

#[cfg(test)]
mod tests {
    use super::*;

    /// A close-to-the-original port of micrograd's sanity check, with hand-derived expectations.
    #[test]
    fn sanity_check() {
        // f = a*b + a^2, with a = -4, b = 2
        // df/da = b + 2a = 2 + (-8) = -6 ; df/db = a = -4
        let a = Value::new(-4.0);
        let b = Value::new(2.0);
        let f = a.clone() * b.clone() + a.clone().powf(2.0);
        f.backward();
        assert!((f.data() - 8.0).abs() < 1e-9); // a*b + a^2 = -8 + 16 = 8
        assert!((a.grad() - (-6.0)).abs() < 1e-9);
        assert!((b.grad() - (-4.0)).abs() < 1e-9);
    }

    #[test]
    fn exp_ln_relu_grads() {
        // y = ln(exp(x)) == x, so dy/dx == 1
        let x = Value::new(0.7);
        let y = x.exp().ln();
        y.backward();
        assert!((y.data() - 0.7).abs() < 1e-9);
        assert!((x.grad() - 1.0).abs() < 1e-9);

        // relu blocks the gradient for negative inputs
        let n = Value::new(-2.0);
        let r = n.relu();
        r.backward();
        assert_eq!(r.data(), 0.0);
        assert_eq!(n.grad(), 0.0);
    }

    /// Gradient accumulation when a node is used more than once: y = x + x, dy/dx = 2.
    #[test]
    fn reused_node_accumulates() {
        let x = Value::new(3.0);
        let y = x.clone() + x.clone();
        y.backward();
        assert!((y.data() - 6.0).abs() < 1e-9);
        assert!((x.grad() - 2.0).abs() < 1e-9);
    }
}
