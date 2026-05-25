---
paths:
  - "**/*.rs"
---

# Testing

- **Unit tests** live in a per-module `#[cfg(test)] mod tests` block at the
  bottom of the file they cover (see `engine.rs`, `nn.rs`, `optim.rs`,
  `tokenizer.rs`).
- **Engine correctness** is guarded by finite-difference gradient checks in
  `tests/grad_check.rs`: build an expression, run analytic `backward()`, then
  compare against central differences `(f(x+ε) - f(x-ε)) / 2ε` within `TOL`.
- **Run `cargo test` before pushing** — the pre-push git hook enforces it.
  Use `cargo test --verbose` to mirror CI.
- Prefer asserting on hand-derived expected values (with a small tolerance for
  floats), as the existing tests do, over golden outputs.
