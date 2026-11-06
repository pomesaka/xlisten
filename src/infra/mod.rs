//! The world outside: one module a thing, each implementing the ports of `crate::app::ports` it
//! stands behind. The `unsafe` of the bindings of C is here only (`speech_cpp`, `tap`).

pub mod events_dir;
pub mod inbox;
pub mod microphone;
pub mod speech_cpp;
pub mod tap;
pub mod wav;
pub mod whisper;
