//! Pipelined GPU → CPU read-back helper.
//!
//! Both the per-frame timestamp drain and auto-exposure EV100 read-back
//! used to call `device.poll(WaitIndefinitely)` immediately after
//! `queue.submit`, blocking the CPU until the GPU finished the
//! previous frame's copy. Two such blocking polls per frame can
//! serialise the CPU with the entire GPU queue depth — typically
//! 0.5–4 ms of unnecessary stall.
//!
//! `PipelinedReadback` solves this with a 2-slot ping-pong:
//!
//! 1. **Frame N — submit:** the caller has just queued the work that
//!    writes data into the slot at `write_slot`. They call
//!    [`PipelinedReadback::submit`] which initiates a non-blocking
//!    `map_async` against that slot and flips `write_slot`.
//! 2. **Frame N+1 — poll + read:** the caller calls
//!    [`PipelinedReadback::try_read`] which calls
//!    `device.poll(PollType::Poll)` (non-blocking, just drains map
//!    callbacks) and, if the slot's `ready` flag has fired, copies
//!    out the bytes and unmaps. Returns `None` if the slot isn't
//!    ready yet — caller falls back to its previous value.
//!
//! In steady state the read is always exactly one frame behind the
//! write — at 60 Hz the GPU has ~16 ms to finish the copy, which is
//! orders of magnitude more than it actually needs. The CPU never
//! blocks waiting.
//!
//! On the first frame the read slot has never been written, so
//! `try_read` returns `None` cleanly without trying to unmap. Same
//! during the brief window after `clear` (used by `App::reconfigure`
//! when pass layout changes).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// State of one ping-pong staging slot.
enum SlotState {
    /// Never written, or already read out and unmapped.
    Idle,
    /// `map_async` initiated, awaiting GPU completion. The slot is
    /// kept here until the next `try_read` finds `ready == true`.
    InFlight {
        ready: Arc<AtomicBool>,
    },
}

struct ReadbackSlot {
    buffer: wgpu::Buffer,
    state: SlotState,
}

/// Two-slot ping-pong staging buffer manager. The GPU writes into the
/// slot pointed at by `write_slot`; the CPU reads from the *other*
/// slot one frame later.
///
/// `BYTES` is the per-slot capacity. The buffer is created with
/// `COPY_DST | MAP_READ`; callers issue
/// `encoder.copy_buffer_to_buffer` into [`PipelinedReadback::write_buffer`]
/// before calling [`submit`](Self::submit).
pub struct PipelinedReadback {
    slots: [ReadbackSlot; 2],
    write_slot: usize,
    /// Per-slot byte capacity. Both slots use the same size.
    byte_capacity: u64,
}

impl PipelinedReadback {
    /// Allocate a new pipelined read-back with two `COPY_DST | MAP_READ`
    /// staging buffers of `byte_capacity` bytes each.
    pub fn new(device: &wgpu::Device, label: &'static str, byte_capacity: u64) -> Self {
        let make_slot = || ReadbackSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: byte_capacity,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            state: SlotState::Idle,
        };
        Self {
            slots: [make_slot(), make_slot()],
            write_slot: 0,
            byte_capacity,
        }
    }

    /// Buffer that the *current* frame's GPU work should write into
    /// via `encoder.copy_buffer_to_buffer(..., dst = this_buffer)`.
    pub fn write_buffer(&self) -> &wgpu::Buffer {
        &self.slots[self.write_slot].buffer
    }

    /// After `queue.submit(...)` the caller calls this to initiate the
    /// non-blocking map of the slot just written. Flips `write_slot`
    /// so the next frame writes the opposite buffer.
    ///
    /// Idempotent in the sense that if the slot is already in flight
    /// (shouldn't normally happen — would mean a previous frame's
    /// read never drained) this drops the old `ready` flag and starts
    /// fresh.
    pub fn submit(&mut self) {
        let slot = &mut self.slots[self.write_slot];
        let ready = Arc::new(AtomicBool::new(false));
        let ready_cb = Arc::clone(&ready);
        slot.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            // We only signal `ready = true` on success. On failure
            // the slot stays InFlight forever — `try_read` will keep
            // returning None. The caller is expected to recover via
            // `clear()` if it suspects a stuck slot.
            if result.is_ok() {
                ready_cb.store(true, Ordering::Release);
            }
        });
        slot.state = SlotState::InFlight { ready };
        self.write_slot ^= 1;
    }

    /// Non-blocking check: if the slot that was submitted *last*
    /// frame is ready, copy its bytes out, unmap, and return
    /// `Some(bytes)`. Returns `None` if the slot is still in flight
    /// or has never been written.
    ///
    /// The "last frame" slot is the one [`submit`] is about to write
    /// *next*: after the most recent `submit` flipped `write_slot`,
    /// the new `write_slot` index points at the slot that was
    /// submitted two frames ago and has had a full GPU-queue length
    /// of time to complete. (Two frames ago = the previous time we
    /// occupied this index.) On the first run after `clear` or
    /// construction, both slots are Idle and this returns `None`
    /// cleanly.
    ///
    /// Calls `device.poll(PollType::Poll)` once to drain any pending
    /// map callbacks before checking the ready flag.
    ///
    /// The index returned alongside the bytes lets the caller pair
    /// per-slot metadata (e.g. pass-name lists) that was captured at
    /// submit time.
    pub fn try_read(&mut self, device: &wgpu::Device) -> Option<(usize, Vec<u8>)> {
        // Drain any pending map callbacks so this frame's `ready`
        // flags can flip.
        let _ = device.poll(wgpu::PollType::Poll);
        let read_slot_idx = self.write_slot;
        let slot = &mut self.slots[read_slot_idx];
        let ready = match &slot.state {
            SlotState::Idle => return None,
            SlotState::InFlight { ready } => ready.load(Ordering::Acquire),
        };
        if !ready {
            return None;
        }
        let bytes = slot.buffer.slice(..self.byte_capacity).get_mapped_range().to_vec();
        slot.buffer.unmap();
        slot.state = SlotState::Idle;
        Some((read_slot_idx, bytes))
    }

    /// Index of the slot that the next [`submit`] will mark in-flight.
    /// Per-slot metadata captured at this index will be paired with
    /// the bytes returned by a subsequent `try_read` once the GPU
    /// completes the copy. Useful when the caller wants to stash
    /// frame-time state (e.g. a pass-name list) alongside the
    /// in-flight slot.
    pub fn write_slot(&self) -> usize {
        self.write_slot
    }

    /// Drop all in-flight state. Call when the upstream pipeline
    /// changes (different pass count, framebuffer resize) — the next
    /// frame's submit starts fresh.
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            if matches!(slot.state, SlotState::InFlight { .. }) {
                // The buffer might still be mapped on the GPU side;
                // unmap defensively. wgpu tolerates double-unmap with
                // a no-op + warn, which is acceptable here.
                slot.buffer.unmap();
            }
            slot.state = SlotState::Idle;
        }
        self.write_slot = 0;
    }

    /// Byte capacity of each slot.
    pub fn byte_capacity(&self) -> u64 {
        self.byte_capacity
    }
}
