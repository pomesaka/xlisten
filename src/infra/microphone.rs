//! The microphone, through cpal (Core Audio). Dropping it stops it.

use std::sync::mpsc::SyncSender;

use anyhow::{Context as _, Result, anyhow, bail};
use cpal::traits::{DeviceTrait as _, HostTrait as _, StreamTrait as _};

use crate::domain::time::Millis;
use crate::domain::utterance::AudioBlock;
use crate::dsp::to_mono;

pub struct Microphone {
    _stream: cpal::Stream,
    pub name: String,
    pub rate: u32,
}

impl Microphone {
    /// The input device named `device` (the default one where `None`), handing each block to `to`.
    ///
    /// # Errors
    /// When there is no such device, or it gives no float samples.
    pub fn start(device: Option<&str>, to: SyncSender<AudioBlock>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match device {
            None => host.default_input_device().context("no microphone")?,
            Some(wanted) => host
                .input_devices()?
                .find(|d| d.description().is_ok_and(|desc| desc.name() == wanted))
                .with_context(|| format!("no microphone named {wanted}"))?,
        };
        let name = device
            .description()
            .map(|d| d.name().to_owned())
            .unwrap_or_default();
        let config = device.default_input_config()?;
        if config.sample_format() != cpal::SampleFormat::F32 {
            bail!("{name} gives {:?}, not f32", config.sample_format());
        }
        let channels = usize::from(config.channels());
        let rate = config.sample_rate();
        let stream = device.build_input_stream(
            config.config(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // WHY try_send: the callback must not wait; a block dropped when the reader is
                // behind is lost.
                let _ = to.try_send(AudioBlock {
                    samples: to_mono(data, channels),
                    end: Millis::now(),
                });
            },
            |e| eprintln!("xlisten: microphone: {e}"),
            None,
        )?;
        stream.play().map_err(|e| anyhow!("{e}"))?;
        Ok(Self {
            _stream: stream,
            name,
            rate,
        })
    }
}
