use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

lazy_static! {
    static ref COUNTERS: Mutex<HashMap<&'static str, u128>> = {
        let counters = HashMap::<&'static str, u128>::new();
        Mutex::new(counters)
    };
}

pub struct Counter {
    name: &'static str,
    start: Instant,
}

impl Counter {
    pub fn new(name: &'static str) -> Self {
        Self { name, start: Instant::now() }
    }
}

impl Drop for Counter {
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_micros();
        let mut counters = COUNTERS.lock().unwrap();

        if let Some(value) = counters.get_mut(&self.name) {
            *value += duration;
        } else {
            counters.insert(self.name, duration);
        }
    }
}

pub struct CounterResult {
    name: &'static str,
    duration: u128,
}

impl CounterResult {
    pub fn new(name: &'static str, duration: u128) -> Self {
        Self { name, duration }
    }
}

pub fn dump_counters() {
    let counters = COUNTERS.lock().unwrap();

    let mut results = Vec::<CounterResult>::new();

    for (name, micros) in counters.iter() {
        let duration = micros / 1000;
        results.push(CounterResult::new(name, duration))
    }

    results.sort_by(|a, b| b.duration.cmp(&a.duration));

    for result in results.iter() {
        println!("{}: {}ms", result.name, result.duration);
    }
}
