// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//! Router-local in-flight input (and optionally streamed output) tokens, not
//! physical allocator occupancy.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::Value;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    #[default]
    Off,
    Shadow,
    On,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Demand {
    pub tokens: u128,
    pub requests: usize,
}
pub(crate) type Ledger = Arc<Mutex<HashMap<String, Demand>>>;

/// Text bytes per token assumed when a request's tokens are unavailable.
const ESTIMATED_BYTES_PER_TOKEN: usize = 4;

/// Input tokens estimated from a body's string bytes. Inline binary payloads
/// (`data:` URLs and `data` fields, such as base64 images or audio) are
/// skipped: their size says nothing about their tokens.
pub(crate) fn estimate_tokens(body: &Value) -> u64 {
    text_bytes(body).div_ceil(ESTIMATED_BYTES_PER_TOKEN) as u64
}

fn text_bytes(value: &Value) -> usize {
    match value {
        Value::String(s) if s.starts_with("data:") => 0,
        Value::String(s) => s.len(),
        Value::Array(items) => items.iter().map(text_bytes).sum(),
        Value::Object(fields) => fields
            .iter()
            .filter(|(key, _)| key.as_str() != "data")
            .map(|(_, value)| text_bytes(value))
            .sum(),
        _ => 0,
    }
}

/// Longest partial SSE line carried across chunks; a longer one is dropped.
const MAX_PARTIAL_LINE: usize = 1 << 16;

/// Output tokens estimated from a decode SSE stream: the text bytes of each
/// event's generated part (`choices[].delta` / `choices[].text`, or a
/// Responses `*.delta` event's `delta`), at [`ESTIMATED_BYTES_PER_TOKEN`].
/// Engines batch tokens per event (vLLM `--stream-interval`), so events are
/// not tokens.
#[derive(Debug, Default)]
pub(crate) struct OutputMeter {
    partial: Vec<u8>,
    bytes: usize,
    counted: u64,
}

impl OutputMeter {
    /// Tokens the chunk adds; a line split across chunks counts once complete.
    pub(crate) fn feed(&mut self, chunk: &[u8]) -> u64 {
        self.partial.extend_from_slice(chunk);
        let buf = std::mem::take(&mut self.partial);
        let mut lines: Vec<&[u8]> = buf.split(|&b| b == b'\n').collect();
        let tail = lines.pop().unwrap_or_default();
        for line in lines {
            let Some(data) = line.trim_ascii().strip_prefix(b"data:") else {
                continue;
            };
            if let Ok(event) = serde_json::from_slice::<Value>(data) {
                self.bytes += generated_bytes(&event);
            }
        }
        if tail.len() <= MAX_PARTIAL_LINE {
            self.partial = tail.to_vec();
        }
        let total = self.bytes.div_ceil(ESTIMATED_BYTES_PER_TOKEN) as u64;
        let added = total - self.counted;
        self.counted = total;
        added
    }
}

fn generated_bytes(event: &Value) -> usize {
    if let Some(choices) = event.get("choices").and_then(Value::as_array) {
        return choices
            .iter()
            .filter_map(|c| c.get("delta").or_else(|| c.get("text")))
            .map(text_bytes)
            .sum();
    }
    let is_delta = event
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|t| t.ends_with(".delta"));
    match event.get("delta") {
        Some(delta) if is_delta => text_bytes(delta),
        _ => 0,
    }
}

/// Booked atomically with selection and released even if the pick is abandoned.
#[derive(Debug)]
pub struct Reservation {
    ledger: Ledger,
    key: String,
    tokens: u64,
    // Some when streamed output is charged too (`--decode-output-demand`).
    output: Option<OutputMeter>,
}

impl Reservation {
    pub(crate) fn book(
        ledger: &Ledger,
        state: &mut HashMap<String, Demand>,
        key: String,
        tokens: u64,
    ) -> Self {
        let entry = state.entry(key.clone()).or_default();
        entry.tokens += u128::from(tokens);
        entry.requests += 1;
        crate::metrics::set_decode_demand_tokens(&key, Some(entry.tokens));
        crate::metrics::set_decode_demand_requests(&key, Some(entry.requests));
        Self {
            ledger: ledger.clone(),
            key,
            tokens,
            output: None,
        }
    }

    /// Also charge the output the decode streams back, until release.
    pub(crate) fn tracking_output(mut self, on: bool) -> Self {
        self.output = on.then(OutputMeter::default);
        self
    }

    /// Add the output tokens in a streamed chunk. No-op unless tracking output.
    pub(crate) fn observe_output(&mut self, chunk: &[u8]) {
        let Some(meter) = self.output.as_mut() else {
            return;
        };
        let added = meter.feed(chunk);
        if added == 0 {
            return;
        }
        let mut state = self.ledger.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(entry) = state.get_mut(&self.key) else {
            return;
        };
        entry.tokens += u128::from(added);
        self.tokens = self.tokens.saturating_add(added);
        crate::metrics::set_decode_demand_tokens(&self.key, Some(entry.tokens));
    }
}

// Never panics: drop can run while unwinding, where a second panic aborts the router.
impl Drop for Reservation {
    fn drop(&mut self) {
        let mut state = self.ledger.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(entry) = state.get_mut(&self.key) else {
            tracing::warn!(route_key = %self.key, "decode demand released without a reservation");
            return;
        };
        entry.tokens = entry.tokens.saturating_sub(u128::from(self.tokens));
        entry.requests = entry.requests.saturating_sub(1);
        if entry.requests == 0 {
            state.remove(&self.key);
            crate::metrics::set_decode_demand_tokens(&self.key, None);
            crate::metrics::set_decode_demand_requests(&self.key, None);
        } else {
            crate::metrics::set_decode_demand_tokens(&self.key, Some(entry.tokens));
            crate::metrics::set_decode_demand_requests(&self.key, Some(entry.requests));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_inputs_add_and_release_independently() {
        let ledger = Ledger::default();
        let (a, b) = {
            let mut state = ledger.lock().unwrap();
            let a = Reservation::book(&ledger, &mut state, "d#dp0".into(), 32000);
            let b = Reservation::book(&ledger, &mut state, "d#dp0".into(), 32000);
            assert_eq!(
                state["d#dp0"],
                Demand {
                    tokens: 64000,
                    requests: 2,
                }
            );
            (a, b)
        };
        drop(a);
        assert_eq!(ledger.lock().unwrap()["d#dp0"].tokens, 32000);
        drop(b);
        assert!(ledger.lock().unwrap().is_empty());
    }

    #[test]
    fn large_reservations_do_not_overflow_or_lose_small_charges() {
        let ledger = Ledger::default();
        let (a, b, small) = {
            let mut state = ledger.lock().unwrap();
            (
                Reservation::book(&ledger, &mut state, "d".into(), u64::MAX),
                Reservation::book(&ledger, &mut state, "d".into(), u64::MAX),
                Reservation::book(&ledger, &mut state, "d".into(), 1),
            )
        };
        assert_eq!(
            ledger.lock().unwrap()["d"].tokens,
            u128::from(u64::MAX) * 2 + 1
        );
        drop((a, b));
        assert_eq!(ledger.lock().unwrap()["d"].tokens, 1);
        drop(small);
        assert!(ledger.lock().unwrap().is_empty());
    }

    #[test]
    fn release_tolerates_a_missing_entry_and_a_poisoned_ledger() {
        let ledger = Ledger::default();
        let (gone, kept) = {
            let mut state = ledger.lock().unwrap();
            (
                Reservation::book(&ledger, &mut state, "gone".into(), 5),
                Reservation::book(&ledger, &mut state, "kept".into(), 7),
            )
        };
        ledger.lock().unwrap().remove("gone");
        drop(gone);
        let poisoner = ledger.clone();
        let _ = std::thread::spawn(move || {
            let _held = poisoner.lock().unwrap();
            panic!("poison the ledger");
        })
        .join();
        assert!(ledger.is_poisoned());
        drop(kept);
        assert!(ledger
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty());
    }

    #[test]
    fn booked_tokens_are_exported_until_released() {
        let series = "infera_router_decode_demand_tokens{target=\"metrics-test#dp1\"}";
        let exported = || {
            let text = String::from_utf8(crate::metrics::render().0).unwrap();
            text.lines()
                .find_map(|line| line.strip_prefix(series).map(str::trim).map(str::to_owned))
        };
        let ledger = Ledger::default();
        let reservation = {
            let mut state = ledger.lock().unwrap();
            Reservation::book(&ledger, &mut state, "metrics-test#dp1".into(), 5)
        };
        assert_eq!(exported().as_deref(), Some("5"));
        drop(reservation);
        assert_eq!(exported(), None);
    }

    #[test]
    fn estimates_count_text_bytes_and_skip_inline_data() {
        let body = serde_json::json!({"messages":[
            {"role":"user","content":[
                {"type":"text","text":"12345678"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},
                {"type":"input_audio","input_audio":{"data":"UklGRiQAAABXQVZF","format":"wav"}},
            ]},
        ]});
        // "user" + "text" + "12345678" + "image_url" + "input_audio" + "wav"
        // = 39 bytes -> 10 tokens.
        assert_eq!(estimate_tokens(&body), 10);
        assert_eq!(estimate_tokens(&serde_json::json!({})), 0);
    }

    #[test]
    fn output_meter_counts_generated_text_across_split_chunks() {
        let mut meter = OutputMeter::default();
        // Only the delta counts: "12345678" is 8 bytes -> 2 tokens.
        let event = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\
                     \"choices\":[{\"index\":0,\"delta\":{\"content\":\"12345678\"}}]}\n\n";
        let (head, tail) = event.split_at(20);
        assert_eq!(meter.feed(head.as_bytes()), 0);
        assert_eq!(meter.feed(tail.as_bytes()), 2);
        let completion = "data: {\"choices\":[{\"text\":\"abcd\"}]}\n\n\
                          data: {\"type\":\"response.output_text.delta\",\"delta\":\"efgh\"}\n\n\
                          data: {\"type\":\"response.created\",\"response\":{\"id\":\"r\"}}\n\n\
                          data: [DONE]\n\n";
        assert_eq!(meter.feed(completion.as_bytes()), 2);
        assert_eq!(meter.feed(b": keep-alive\n\n"), 0);
    }

    #[test]
    fn tracked_output_grows_the_charge_until_release() {
        let ledger = Ledger::default();
        let chunk = b"data: {\"choices\":[{\"delta\":{\"content\":\"0123456789abcdef\"}}]}\n\n";
        let mut untracked = {
            let mut state = ledger.lock().unwrap();
            Reservation::book(&ledger, &mut state, "d#dp0".into(), 100)
        };
        untracked.observe_output(chunk);
        assert_eq!(ledger.lock().unwrap()["d#dp0"].tokens, 100);
        let mut tracked = {
            let mut state = ledger.lock().unwrap();
            Reservation::book(&ledger, &mut state, "d#dp0".into(), 100).tracking_output(true)
        };
        tracked.observe_output(chunk);
        tracked.observe_output(chunk);
        assert_eq!(ledger.lock().unwrap()["d#dp0"].tokens, 208);
        drop(tracked);
        assert_eq!(
            ledger.lock().unwrap()["d#dp0"],
            Demand {
                tokens: 100,
                requests: 1,
            }
        );
        drop(untracked);
        assert!(ledger.lock().unwrap().is_empty());
    }
}
