//! The use cases: in which order things happen, on which thread, and what a failure does. They
//! decide nothing of the meeting themselves (`crate::domain` does), and reach the world outside
//! only through `ports`.

pub mod latency;
pub mod listen;
pub mod pipeline;
pub mod ports;
pub mod transcribe;

/// A line for the person on stderr. The lines for the agent are `Event`s.
pub fn log(message: &str) {
    eprintln!("xlisten: {message}");
}
