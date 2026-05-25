---
paths:
  - "src/engine.rs"
  - "src/nn.rs"
  - "tests/grad_check.rs"
---

# Autograd invariant

The engine's `backward()` applies one uniform rule to every node:
`child.grad += local_grad * upstream` (the chain rule, accumulated over the
graph). Everything downstream depends on this contract.

- **Record children *and* their local gradients at forward time.** When adding
  or modifying a differentiable op or layer, build the result node with
  `Value::from_op(data, children, local_grads)` — compute each local derivative
  (∂result/∂child) during the forward pass. A missing or wrong local grad
  silently breaks training.
- **Accumulate, never overwrite.** The graph is a DAG of `Rc<RefCell<Inner>>`;
  one node can feed many others, so gradients must use `+=`. Forward passes
  build fresh nodes (grads start at zero), which is why `backward()` assumes a
  clean slate.
- **Add a gradient check.** Any new op gets a finite-difference test in
  `tests/grad_check.rs` comparing analytic gradients to central differences.
- Keep local-gradient math beside the op with a comment deriving it (e.g.
  `d(a*b)/da = b`), matching the existing style.
