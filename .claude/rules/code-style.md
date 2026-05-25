---
paths:
  - "**/*.rs"
---

# Code style

- **No tensor abstraction.** Everything operates on `Value` / `Vec<Value>` /
  `&[Value]`. Each scalar is its own node in the computation graph; that is the
  whole point. Don't introduce tensors, ndarrays, or batching.
- **Clone `Value` to reuse an operand.** Clones are cheap (a refcount bump) and
  share the same underlying node, so a gradient written through one clone is
  visible through all of them. Reusing an operand (e.g. a residual) is an
  explicit `.clone()`.
- **Match the teaching-doc density.** Write `//!` module docs and `///` item
  docs that explain the math and the *why*, not just the *what*. Where a piece
  ports Karpathy's original, say so. This repo is read to be learned from.
- **Formatting and lint are non-negotiable.** Code must pass `cargo fmt` and
  `cargo clippy --all-targets -- -D warnings` (clippy warnings are errors).
