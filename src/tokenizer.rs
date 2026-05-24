//! A minimal character-level tokenizer.
//!
//! It assigns each distinct character in the dataset an integer id `0..n`, and reserves one extra
//! id for a special **beginning/end-of-sequence** marker (`BOS`). Surrounding every document with
//! `BOS` lets the model learn where names tend to start and stop.

use std::collections::BTreeSet;

/// Translates between characters and integer token ids.
pub struct Tokenizer {
    /// Distinct dataset characters, sorted; index == token id.
    chars: Vec<char>,
}

impl Tokenizer {
    /// Build a tokenizer from the documents, collecting the sorted set of unique characters.
    pub fn from_docs(docs: &[String]) -> Self {
        let set: BTreeSet<char> = docs.iter().flat_map(|d| d.chars()).collect();
        Tokenizer {
            chars: set.into_iter().collect(),
        }
    }

    /// The BOS token id (one past the last real character).
    pub fn bos(&self) -> usize {
        self.chars.len()
    }

    /// Total number of tokens, including BOS.
    pub fn vocab_size(&self) -> usize {
        self.chars.len() + 1
    }

    /// Map a character to its token id. Panics if the character was not seen at build time.
    pub fn encode(&self, ch: char) -> usize {
        self.chars
            .iter()
            .position(|&c| c == ch)
            .expect("character not in vocabulary")
    }

    /// Map a token id back to its character, or `None` for BOS.
    pub fn decode(&self, id: usize) -> Option<char> {
        self.chars.get(id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_sorted_vocab_with_bos() {
        let docs = vec!["cab".to_string(), "ba".to_string()];
        let tok = Tokenizer::from_docs(&docs);
        // unique sorted chars: a, b, c -> ids 0,1,2 ; BOS = 3 ; vocab = 4
        assert_eq!(tok.vocab_size(), 4);
        assert_eq!(tok.bos(), 3);
        assert_eq!(tok.encode('a'), 0);
        assert_eq!(tok.encode('c'), 2);
        assert_eq!(tok.decode(1), Some('b'));
        assert_eq!(tok.decode(tok.bos()), None);
    }
}
