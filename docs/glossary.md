# Glossary

The words of a meeting's transcript, as the person and the agent use them, and the names in the code.

| Word | 日本語 | In the code | Meaning |
|---|---|---|---|
| me | 自分 | `Input::Me` | What the microphone heard: the person who runs xlisten, and in a meeting held in person, the room |
| others | 相手 | `Input::Others` | What the apps of the Mac played: the others of an online meeting, and whatever else played |
| utterance | 発話 | `Utterance` | What one input heard from where somebody starts speaking to where they stop |
| speaker | 話者 | `SpeakerLabel` | `A`, `B`, ... among `others`, by voice, in the order first heard; holds for one run only, and is not a name |
| transcript | 文字起こし | `Transcript`, `Event::Speech` | The words of each utterance, numbered in the order they were read |
| note | メモ | `Event::Note` | A line added to `inbox.txt`, a mark among the transcript |
| minutes | 議事録 | (none) | What the agent writes from the transcript when asked |
