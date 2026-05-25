# rustynet

A modern neural network built from scratch in Rust, for **learning** — not speed.

It is inspired by Andrej Karpathy's "most atomic" pure-Python GPT, but deliberately strips away the
transformer machinery (attention, KV-cache, transformer positional embeddings). What's left is the
part that is *the algorithm*: a scalar reverse-mode **autograd engine**, a few **layers**,
**normalization**, a **softmax cross-entropy** loss, and the **Adam** optimizer. These are wired
into a Bengio-style character-level MLP language model — the historical predecessor of GPT — that
learns to babble brand-new names.

## Run it

```sh
# install the git hooks once per clone
prek install

# fetch the dataset (Karpathy's list of names) once
mkdir -p data && curl -L -o data/names.txt \
  https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt

cargo run --release
```

You'll see the training loss fall, then 20 hallucinated names like `karie`, `jamian`, `kameri`.

```sh
cargo test    # unit tests + finite-difference gradient checks
```

## How it fits together

| File | Building block |
|------|----------------|
| `src/engine.rs` | `Value`: the scalar autograd engine (forward graph + `backward()`) |
| `src/nn.rs` | `Linear`, `Embedding`, `rmsnorm`, `relu`, `softmax`, `cross_entropy`, `Module` trait |
| `src/optim.rs` | `Adam` optimizer with bias correction |
| `src/tokenizer.rs` | character ↔ token-id translation (+ a special BOS token) |
| `src/model.rs` | `NameMLP`: embed context → RMSNorm → Linear → ReLU → Linear → logits |
| `src/main.rs` | the training loop and autoregressive sampling |

The whole point is clarity: every scalar is its own node in the computation graph, so the autograd
engine handles all gradients automatically. Efficiency (tensors, batching, vectorization) is
intentionally left out — that is "everything else" beyond the core algorithm.
