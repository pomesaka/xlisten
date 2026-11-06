use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Which input an utterance came through: `me`, the microphone (the person, and in a meeting held in
/// person, the room), or `others`, the sound the apps of the Mac play.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Input {
    Me,
    Others,
}

impl Input {
    /// Whether RE-USE goes over it before whisper reads it.
    /// WHY the microphone only: RE-USE takes the noise and the echo of a room out of a recording;
    /// the sound of the apps comes as the meeting app sent it, already cleaned.
    #[must_use]
    pub const fn is_enhanced(self) -> bool {
        matches!(self, Self::Me)
    }

    /// Whether its utterances are told apart by voice (`speaker`).
    /// WHY not the microphone: online, it is the person's own voice. A room heard in person is not
    /// told apart (not tried).
    #[must_use]
    pub const fn has_speakers(self) -> bool {
        matches!(self, Self::Others)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Me => "me",
            Self::Others => "others",
        }
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Input {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "me" => Ok(Self::Me),
            "others" => Ok(Self::Others),
            _ => Err(format!("{s} is neither me nor others")),
        }
    }
}
