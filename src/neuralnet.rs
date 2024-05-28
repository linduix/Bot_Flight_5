use serde_json::Value;
use std::collections::HashMap;

fn recursive_search(order_vec: &mut Vec<i64>,
                    dependencies: &HashMap<i64, Vec<(i64, f64)>>,
                    id: i64) {

    // Skip if already in order vec
    if order_vec.contains(&id) {
        return;
    }

    // root nodes are 0-7, add to vec if root then return
    if id <= 7 {
        order_vec.push(id);
        return;
    }

    // get the children of the node if not root
    if let Some(deps) = dependencies.get(&id) {
        // loop over children
        for (child, _) in deps {
            // recurse over children
            recursive_search(order_vec, dependencies, *child)
        }
        // add self to list once children are added
        order_vec.push(id)
    }
}

struct Node {
    bias: f64,
    value: f64,
    activation: fn(f64) -> f64
}

impl Node {
    fn output(&self) -> f64 {
        (self.activation)(self.value + self.bias)
    }

    fn new(bias: f64, layer: i8) -> Node {
        let activation: fn(f64) -> f64;
        match layer {
            0 => { activation = |x| x }
            1 => { activation = |x| x.max(0f64) }
            _ => { activation = |x| x.tanh() }
        }
        Node {bias, activation, value: 0f64}
    }
}

pub struct NeuralNet<'a> {
    pub genotype: &'a Value,
    nodes: HashMap<i64, Node>,
    dependencies: HashMap<i64, Vec<(i64, f64)>>,

    input_ids: [i64; 8],
    output_ids: [i64; 2],
    topology: Vec<i64>
}
impl<'a> NeuralNet<'a> {
    pub fn new(gene: &'a Value) -> NeuralNet {
        let nodes_data: &Value = gene.get("nodes").unwrap();
        let connections_data: &Value = gene.get("connections").unwrap();

        // ID : Node struct
        let mut nodes: HashMap<i64, Node> = HashMap::new();
        // Output Node : Vec[(Input Node: Weight), ...]
        let mut dependencies: HashMap<i64, Vec<(i64, f64)>> = HashMap::new();

        // Node ID vectors for activation order
        let mut topology: Vec<i64> = vec![];

        // Decode nodes data
        if let Value::Object(map) = nodes_data {
            for (key, value) in map {
                let id = key.parse::<i64>().unwrap();
                let layer = value.get("layer").unwrap().as_i64().unwrap();
                let bias: f64;
                match layer {
                    // If node in input layer (0), bias is set to 0
                    0 => {bias = 0f64},
                    _ => {bias = value.get("bias").unwrap().as_f64().unwrap();}
                }
                // Add decoded data to hashmap
                nodes.insert(id, Node::new(bias, layer as i8));
            }
        }

        // Decode connections data
        if let Value::Object(map) = connections_data {
            let pattern = |c| c=='(' || c==')';
            for (key, value) in map {
                let enabled = value.get("enabled").unwrap().as_bool().unwrap();

                if enabled {
                    // decode str to edge vector
                    let edge: Vec<&str> = key.trim_matches(pattern).split(',').collect();
                    let weight =  value.get("weight").unwrap().as_f64().unwrap();

                    if let [first, second] =  edge.as_slice() {
                        let inp: i64 = first.trim().parse().unwrap();
                        let outp: i64 = second.trim().parse().unwrap();

                        // Get vector or create new if not exist
                        // Then add decoded data to dependency vector
                        dependencies.entry(outp).or_insert_with(Vec::new)
                            .push((inp, weight));
                    }
                }
            }
        }

        // Sort topological activation order
        let outputs: [i64; 2] = [8, 9];
        for output in outputs {
            recursive_search(&mut topology, &dependencies, output)
        }

        NeuralNet {
            genotype: gene,
            nodes,
            dependencies,
            input_ids: [0, 1, 2, 3, 4, 5, 6, 7],
            output_ids: [8, 9],
            topology
        }
    }

    pub fn forward(&mut self, input: &[f64; 8]) -> [f64; 2] {
        // Loop over inputs
        for id in self.input_ids {
            // Get node from hashtable
            if let Some(node) = self.nodes.get_mut(&id) {
                node.value = input[id as usize];
            }
        }

        // Loop over topology
        for id in &self.topology {

            // Get dependencies from Hashmap
            if let Some(deps) = self.dependencies.get(&id) {

                // Loop over dependencies
                let mut activations = 0f64;
                for (d_id, weight) in deps {
                    // Sum all activations
                    activations += self.nodes.get(&d_id).unwrap().output() * weight
                }

                // Get node from hashtable
                if let Some(node) = self.nodes.get_mut(&id) {
                    // Set node value to activation
                    node.value = activations;
                }
            }
        }

        let mut output = [0f64, 0f64];
        // Get values from output
        for i in 0..2 {
            let outp_node: &Node = self.nodes.get( &self.output_ids[i] ).unwrap();
            output[i] = outp_node.output()
        }

        return output
    }
}