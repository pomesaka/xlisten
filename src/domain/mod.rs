//! What a transcript of a meeting is (`docs/adr/0001-domain-model-and-layers.md`): no I/O, and
//! nothing of `app` or `infra`.

pub mod event;
pub mod input;
pub mod speaker;
pub mod time;
pub mod transcript;
pub mod utterance;
