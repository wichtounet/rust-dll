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

pub fn dump_counters_pretty() {
    let counters = COUNTERS.lock().unwrap();

    let mut results = Vec::<CounterResult>::new();

    let mut max_duration: u128 = 0;

    for (name, (micros, count)) in counters.iter() {
        let duration = *micros / 1000;
        results.push(CounterResult::new(name, duration, *count));

        max_duration = std::cmp::max(max_duration, duration);
    }

    results.sort_by(|a, b| b.duration.cmp(&a.duration));

    let mut rows = Vec::new();

    let headers = vec![
        "%".to_string(),
        "Timer".to_string(),
        "Count".to_string(),
        "Total".to_string(),
        "Average".to_string(),
    ];
    rows.push(headers);

    for result in results.iter() {
        let row = vec![
            format!("{:.2}", 100.0 * (result.duration as f64 / max_duration as f64)),
            result.name.to_string(),
            result.count.to_string(),
            format!("{}ms", result.duration),
            format!("{}ms", result.duration & result.count),
        ];
        rows.push(row);
    }

    let mut widths = Vec::<usize>::new();

    for row in &rows {
        for (i, column) in row.iter().enumerate() {
            if widths.len() > i {
                widths[i] = std::cmp::max(widths[i], column.len());
            } else {
                widths.push(column.len());
            }
        }
    }

    let max_width = 1 + widths.iter().sum::<usize>() + 3 * widths.len();

    for (i, row) in rows.iter().enumerate() {
        if i == 0 {
            println!("{}", "-".repeat(max_width));
        }

        print!("| ");

        for (i, column) in row.iter().enumerate() {
            print!("{:1$} | ", column, widths[i]);
        }

        println!();

        if i == 0 || i == rows.len() - 1 {
            println!("{}", "-".repeat(max_width));
        }
    }
}
