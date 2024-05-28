use std::fs;
use std::fs::File;
use std::io::Write;
use serde_json::{json, Value};
use crate::drone::Drone;
use crate::level::run_level;
use rayon::prelude::*;

mod neuralnet;
mod drone;
mod level;

fn main() {
    // Load data
    let file_content = fs::read_to_string("./data/drones.json").unwrap();
    let data: Value = serde_json::from_str(&file_content).unwrap();

    let mut drones: Vec<Drone> = vec![];
    if let Some(array) = data.as_array() {
        for gene in array {
            // print!("{:#?}", gene.to_string());
            drones.push(Drone::new([1600./40.; 2], gene))
        }
    }

    // Run drones Parallel:
    let size = (drones.len() / 10).max(1);
    drones.par_chunks_mut(size).for_each(run_level);

    print!("done running");

    // Save data
    let mut save_data = Vec::new();
    for d in drones {
        let entry = json!({
            "drone": d.brain.genotype,
            "score": d.score,
            "crash": d.crash,
            "completed": d.completed,
            "survived": d.survived,
        });
        save_data.push(entry)
    }

    let jsonized = json!(save_data);
    let json_str = jsonized.to_string();

    let mut file = File::create("data/results.json").expect("Error making file");
    file.write_all(json_str.as_bytes()).expect("Error while writing data");
}
