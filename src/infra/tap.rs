//! The sound every app of this Mac plays, through a process tap of Core Audio (macOS 14.2 and
//! later): a mono global tap, in a private aggregate device whose `IOProc` hands its samples on.
//! Apple's way ("Capturing system audio with Core Audio taps"), and catap's, which the prototype in
//! Python used. The one module that calls Core Audio, so the `unsafe` of it is here.
//!
//! WHY every process and not the meeting app's: the global tap catches whichever process of Slack
//! (Electron) plays a huddle; a Slack huddle came through it on 2026-11-05 (Meet and Zoom not
//! run yet). Whatever else plays (music, a video) is transcribed as `others` too.
#![allow(unsafe_code, reason = "the bindings of Core Audio")]

use std::ffi::{CStr, c_void};
use std::ptr::NonNull;
use std::sync::mpsc::SyncSender;

use anyhow::{Result, bail};
use objc2::AnyThread as _;
use objc2::rc::Retained;
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

use crate::domain::time::Millis;
use crate::domain::utterance::AudioBlock;
use crate::dsp::to_mono;

/// The tap of every process, in an aggregate device of its own, running. Dropping it stops it.
pub struct SystemAudio {
    // WHY in this order: fields are dropped in the order they are declared, the reverse of how
    // they were made. The device stops, its IOProc goes (and with it the Client the IOProc reads),
    // then the device, then the tap.
    _running: Running,
    _io_proc: IoProc,
    _device: AggregateDevice,
    _tap: ProcessTap,
    pub rate: u32,
}

impl SystemAudio {
    /// Taps every process's output, mixed to mono, and hands each block to `to`.
    ///
    /// # Errors
    /// When Core Audio refuses the tap or the device. Without the permission to record the
    /// system's sound, macOS gives a tap that hands only zeros, and no error.
    pub fn start(to: SyncSender<AudioBlock>) -> Result<Self> {
        // Each made is undone by its Drop, should a later one fail.
        let (tap, uid) = ProcessTap::create()?;
        let device = AggregateDevice::create(&uid)?;
        let format = tap.format()?;
        if format.mFormatFlags & kAudioFormatFlagIsFloat == 0 || format.mBitsPerChannel != 32 {
            bail!("the tap gives {format:?}, not float32");
        }
        let channels = usize::try_from(format.mChannelsPerFrame.max(1)).unwrap_or(1);
        let io_proc = IoProc::create(device.0, Box::new(Client { to, channels }))?;
        let running = Running::start(device.0, io_proc.id)?;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a sample rate fits in u32"
        )]
        let rate = format.mSampleRate as u32;
        Ok(Self {
            _running: running,
            _io_proc: io_proc,
            _device: device,
            _tap: tap,
            rate,
        })
    }
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

fn cfstr(s: &CStr) -> CFRetained<CFString> {
    CFString::from_str(&s.to_string_lossy())
}

struct ProcessTap(AudioObjectID);

impl ProcessTap {
    /// A private mono tap of every process, and its UID.
    fn create() -> Result<(Self, CFRetained<CFString>)> {
        // SAFETY: an empty list of processes to leave out is valid.
        let description: Retained<CATapDescription> = unsafe {
            CATapDescription::initMonoGlobalTapButExcludeProcesses(
                CATapDescription::alloc(),
                &NSArray::<NSNumber>::new(),
            )
        };
        // SAFETY: a plain setter of the description.
        unsafe { description.setPrivate(true) };
        let mut tap: AudioObjectID = 0;
        // SAFETY: the description is valid; `tap` is a local the ID is written into.
        ok(
            unsafe { AudioHardwareCreateProcessTap(Some(&description), &raw mut tap) },
            "creating the tap",
        )?;
        // SAFETY: a plain getter of the description.
        let uid = CFString::from_str(&unsafe { description.UUID() }.UUIDString().to_string());
        Ok((Self(tap), uid))
    }

    fn format(&self) -> Result<AudioStreamBasicDescription> {
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
        // SAFETY: `format` has room for `size` bytes, the size of what the property holds.
        ok(
            unsafe {
                AudioObjectGetPropertyData(
                    self.0,
                    NonNull::from(&address),
                    0,
                    std::ptr::null(),
                    NonNull::from(&mut size),
                    NonNull::from(&mut format).cast(),
                )
            },
            "the tap's format",
        )?;
        Ok(format)
    }
}

impl Drop for ProcessTap {
    fn drop(&mut self) {
        // SAFETY: the tap was made by `create`, and no device uses it any more.
        unsafe { AudioHardwareDestroyProcessTap(self.0) };
    }
}

struct AggregateDevice(AudioObjectID);

impl AggregateDevice {
    /// A private device of the tap `tap_uid` alone.
    fn create(tap_uid: &CFString) -> Result<Self> {
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
        // SAFETY: the dictionary is valid; `device` is a local the ID is written into.
        ok(
            unsafe {
                AudioHardwareCreateAggregateDevice(
                    aggregate.as_opaque(),
                    NonNull::from(&mut device),
                )
            },
            "creating the device",
        )?;
        Ok(Self(device))
    }
}

impl Drop for AggregateDevice {
    fn drop(&mut self) {
        // SAFETY: the device was made by `create`, and its IOProc is gone.
        unsafe { AudioHardwareDestroyAggregateDevice(self.0) };
    }
}

/// What the `IOProc` reads on each call.
struct Client {
    to: SyncSender<AudioBlock>,
    channels: usize,
}

struct IoProc {
    device: AudioObjectID,
    id: AudioDeviceIOProcID,
    // WHY after `id`: freed only once the IOProc that reads it is destroyed (in Drop).
    _client: Box<Client>,
}

impl IoProc {
    fn create(device: AudioObjectID, client: Box<Client>) -> Result<Self> {
        let mut id: AudioDeviceIOProcID = None;
        let data: *const Client = &raw const *client;
        // SAFETY: `client` is boxed, so `data` stays where it is until the box is dropped, after
        // the IOProc is destroyed.
        ok(
            unsafe {
                AudioDeviceCreateIOProcID(
                    device,
                    Some(io_proc),
                    data.cast_mut().cast(),
                    NonNull::from(&mut id),
                )
            },
            "the IOProc",
        )?;
        Ok(Self {
            device,
            id,
            _client: client,
        })
    }
}

impl Drop for IoProc {
    fn drop(&mut self) {
        // SAFETY: the IOProc was made by `create`, and the device is stopped.
        unsafe { AudioDeviceDestroyIOProcID(self.device, self.id) };
    }
}

struct Running {
    device: AudioObjectID,
    id: AudioDeviceIOProcID,
}

impl Running {
    fn start(device: AudioObjectID, id: AudioDeviceIOProcID) -> Result<Self> {
        // SAFETY: the device and its IOProc were made by this module.
        ok(
            unsafe { AudioDeviceStart(device, id) },
            "starting the device",
        )?;
        Ok(Self { device, id })
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        // SAFETY: started by `start`.
        unsafe { AudioDeviceStop(self.device, self.id) };
    }
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
    // SAFETY: `client` is the Client of the IoProc, alive until the IOProc is destroyed; `input`
    // is the buffer list Core Audio hands for this call, of float32 frames (checked at start).
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
        // WHY try_send: the IOProc must not wait; a block dropped when the reader is behind is lost.
        let _ = client.to.try_send(AudioBlock {
            samples: to_mono(samples, client.channels),
            end: Millis::now(),
        });
    }
    0
}
