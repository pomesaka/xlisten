//! The sound every app of this Mac plays, through a process tap of Core Audio (macOS 14.2 and
//! later): a mono global tap, in a private aggregate device whose `IOProc` hands its samples on.
//! Apple's way ("Capturing system audio with Core Audio taps"), and catap's, which the prototype in
//! Python used. The one file that calls Core Audio, so the `unsafe` of it is here.
//!
//! WHY every process and not the meeting app's: the global tap catches whichever process of Slack
//! (Electron) plays a huddle; a Slack huddle came through it on 2026-11-05 (Meet and Zoom not
//! run yet). Whatever else
//! plays (music, a video) is transcribed as `others` too.
#![allow(unsafe_code, reason = "the bindings of Core Audio")]

use std::ffi::{CStr, c_void};
use std::ptr::NonNull;
use std::sync::mpsc::SyncSender;

use anyhow::{Result, bail};
use objc2::AnyThread as _;
use objc2_core_audio::{
    AudioDeviceCreateIOProcID, AudioDeviceDestroyIOProcID, AudioDeviceIOProcID, AudioDeviceStart,
    AudioDeviceStop, AudioHardwareCreateAggregateDevice, AudioHardwareCreateProcessTap,
    AudioHardwareDestroyAggregateDevice, AudioHardwareDestroyProcessTap,
    AudioObjectGetPropertyData, AudioObjectID, AudioObjectPropertyAddress, CATapDescription,
    kAudioAggregateDeviceIsPrivateKey, kAudioAggregateDeviceNameKey,
    kAudioAggregateDeviceTapAutoStartKey, kAudioAggregateDeviceTapListKey,
    kAudioAggregateDeviceUIDKey, kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal,
    kAudioSubTapDriftCompensationKey, kAudioSubTapUIDKey, kAudioTapPropertyFormat,
};
use objc2_core_audio_types::{
    AudioBufferList, AudioStreamBasicDescription, AudioTimeStamp, kAudioFormatFlagIsFloat,
};
use objc2_core_foundation::{CFArray, CFBoolean, CFDictionary, CFRetained, CFString, CFType};
use objc2_foundation::{NSArray, NSNumber};

use crate::lines::now_ms;

/// A block of mono samples and the wall clock of its last sample.
pub type Block = (Vec<f32>, f64);

struct Client {
    to: SyncSender<Block>,
    channels: usize,
}

pub struct SystemAudio {
    tap: AudioObjectID,
    device: AudioObjectID,
    proc_id: AudioDeviceIOProcID,
    client: *mut Client,
    pub rate: u32,
}

fn cfstr(s: &CStr) -> CFRetained<CFString> {
    CFString::from_str(&s.to_string_lossy())
}

fn ok(status: i32, what: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let code = status.to_be_bytes();
    let fourcc = if code.iter().all(u8::is_ascii_graphic) {
        format!(" '{}'", String::from_utf8_lossy(&code))
    } else {
        String::new()
    };
    bail!("Core Audio: {what}: {status}{fourcc}")
}

unsafe extern "C-unwind" fn io_proc(
    _device: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    input: NonNull<AudioBufferList>,
    _input_time: NonNull<AudioTimeStamp>,
    _output: NonNull<AudioBufferList>,
    _output_time: NonNull<AudioTimeStamp>,
    client: *mut c_void,
) -> i32 {
    // SAFETY: `client` is the Client of `SystemAudio::start`, alive until the IOProc is destroyed;
    // `input` is the buffer list Core Audio hands for this call, of float32 frames (checked at start).
    unsafe {
        let client = &*client.cast::<Client>();
        let list = input.as_ref();
        if list.mNumberBuffers == 0 {
            return 0;
        }
        let buffer = &list.mBuffers[0];
        if buffer.mData.is_null() {
            return 0;
        }
        let n = usize::try_from(buffer.mDataByteSize).unwrap_or(0) / 4;
        let samples = std::slice::from_raw_parts(buffer.mData.cast::<f32>(), n);
        let mono = crate::audio::to_mono(samples, client.channels);
        // WHY try_send: the IOProc must not wait; a block dropped when the reader is behind is lost.
        #[expect(clippy::cast_precision_loss, reason = "ms since the epoch fit in f64")]
        let _ = client.to.try_send((mono, now_ms() as f64));
    }
    0
}

impl SystemAudio {
    /// Taps every process's output, mixed to mono, and hands each block to `to`.
    ///
    /// # Errors
    /// When Core Audio refuses the tap or the device. Without the permission to record the
    /// system's sound, macOS gives a tap that hands only zeros, and no error.
    #[expect(
        clippy::too_many_lines,
        reason = "one sequence of Core Audio calls, each undone where a later one fails"
    )]
    pub fn start(to: SyncSender<Block>) -> Result<Self> {
        // SAFETY: plain calls of Core Audio with valid arguments; each object made is destroyed in Drop
        // (or here, where a later step fails).
        unsafe {
            let description = CATapDescription::initMonoGlobalTapButExcludeProcesses(
                CATapDescription::alloc(),
                &NSArray::<NSNumber>::new(),
            );
            description.setPrivate(true);
            let mut tap: AudioObjectID = 0;
            ok(
                AudioHardwareCreateProcessTap(Some(&description), &raw mut tap),
                "creating the tap",
            )?;
            let tap_uid = CFString::from_str(&description.UUID().UUIDString().to_string());

            let sub_tap = CFDictionary::<CFString, CFType>::from_slices(
                &[
                    &*cfstr(kAudioSubTapUIDKey),
                    &*cfstr(kAudioSubTapDriftCompensationKey),
                ],
                &[tap_uid.as_ref(), CFBoolean::new(true).as_ref()],
            );
            let taps = CFArray::<CFType>::from_objects(&[sub_tap.as_ref()]);
            let uid = CFString::from_str(&format!("xlisten-{}", std::process::id()));
            let name = CFString::from_str("xlisten");
            let aggregate = CFDictionary::<CFString, CFType>::from_slices(
                &[
                    &*cfstr(kAudioAggregateDeviceNameKey),
                    &*cfstr(kAudioAggregateDeviceUIDKey),
                    &*cfstr(kAudioAggregateDeviceIsPrivateKey),
                    &*cfstr(kAudioAggregateDeviceTapAutoStartKey),
                    &*cfstr(kAudioAggregateDeviceTapListKey),
                ],
                &[
                    name.as_ref(),
                    uid.as_ref(),
                    CFBoolean::new(true).as_ref(),
                    CFBoolean::new(true).as_ref(),
                    taps.as_ref(),
                ],
            );
            let mut device: AudioObjectID = 0;
            if let Err(e) = ok(
                AudioHardwareCreateAggregateDevice(
                    aggregate.as_opaque(),
                    NonNull::from(&mut device),
                ),
                "creating the device",
            ) {
                AudioHardwareDestroyProcessTap(tap);
                return Err(e);
            }

            let mut format = AudioStreamBasicDescription {
                mSampleRate: 0.0,
                mFormatID: 0,
                mFormatFlags: 0,
                mBytesPerPacket: 0,
                mFramesPerPacket: 0,
                mBytesPerFrame: 0,
                mChannelsPerFrame: 0,
                mBitsPerChannel: 0,
                mReserved: 0,
            };
            let mut size = u32::try_from(size_of::<AudioStreamBasicDescription>()).unwrap_or(0);
            let address = AudioObjectPropertyAddress {
                mSelector: kAudioTapPropertyFormat,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain,
            };
            let got = ok(
                AudioObjectGetPropertyData(
                    tap,
                    NonNull::from(&address),
                    0,
                    std::ptr::null(),
                    NonNull::from(&mut size),
                    NonNull::from(&mut format).cast(),
                ),
                "the tap's format",
            );
            let fail = |e: anyhow::Error| {
                AudioHardwareDestroyAggregateDevice(device);
                AudioHardwareDestroyProcessTap(tap);
                Err(e)
            };
            if let Err(e) = got {
                return fail(e);
            }
            if format.mFormatFlags & kAudioFormatFlagIsFloat == 0 || format.mBitsPerChannel != 32 {
                return fail(anyhow::anyhow!("the tap gives {format:?}, not float32"));
            }
            let channels = usize::try_from(format.mChannelsPerFrame.max(1)).unwrap_or(1);
            let client = Box::into_raw(Box::new(Client { to, channels }));
            let mut proc_id: AudioDeviceIOProcID = None;
            if let Err(e) = ok(
                AudioDeviceCreateIOProcID(
                    device,
                    Some(io_proc),
                    client.cast(),
                    NonNull::from(&mut proc_id),
                ),
                "the IOProc",
            ) {
                drop(Box::from_raw(client));
                return fail(e);
            }
            if let Err(e) = ok(AudioDeviceStart(device, proc_id), "starting the device") {
                AudioDeviceDestroyIOProcID(device, proc_id);
                drop(Box::from_raw(client));
                return fail(e);
            }
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a sample rate fits in u32"
            )]
            let rate = format.mSampleRate as u32;
            Ok(Self {
                tap,
                device,
                proc_id,
                client,
                rate,
            })
        }
    }
}

impl Drop for SystemAudio {
    fn drop(&mut self) {
        // SAFETY: the objects were made in `start`; the IOProc is stopped and destroyed before its
        // Client is freed.
        unsafe {
            AudioDeviceStop(self.device, self.proc_id);
            AudioDeviceDestroyIOProcID(self.device, self.proc_id);
            drop(Box::from_raw(self.client));
            AudioHardwareDestroyAggregateDevice(self.device);
            AudioHardwareDestroyProcessTap(self.tap);
        }
    }
}
