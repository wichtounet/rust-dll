use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

lazy_static! {
    static ref COUNTERS: Mutex<HashMap<&'static str, (u128, u128)>> = {
        let counters = HashMap::<&'static str, (u128, u128)>::new();
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

        if let Some((total_duration, count)) = counters.get_mut(&self.name) {
            *total_duration += duration;
            *count += 1;
        } else {
            counters.insert(self.name, (duration, 1));
        }
    }
}

pub struct CounterResult {
    name: &'static str,
    duration: u128,
    count: u128,
}

impl CounterResult {
    pub fn new(name: &'static str, duration: u128, count: u128) -> Self {
        Self { name, duration, count }
    }
}

pub fn dump_counters() {
    let counters = COUNTERS.lock().unwrap();

    let mut results = Vec::<CounterResult>::new();

    for (name, (micros, count)) in counters.iter() {
        let duration = *micros / 1000;
        results.push(CounterResult::new(name, duration, *count))
    }

    results.sort_by(|a, b| b.duration.cmp(&a.duration));

    for result in results.iter() {
        println!("{}: {}ms", result.name, result.duration);
    }
}
