use etl::etl_expr::EtlExpr;
use etl::matrix_2d::Matrix2d;
use etl::vector::Vector;

use crate::layer::Layer;
use crate::table::print_table;

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn get_layer(&mut self, layer: usize) -> &Box<dyn Layer> {
        &self.layers[layer]
    }

    pub fn get_layer_mut(&mut self, layer: usize) -> &mut Box<dyn Layer> {
        &mut self.layers[layer]
    }

    pub fn new_output(&self) -> Vector<f32> {
        self.layers.last().expect("No layers").new_output()
    }

    pub fn new_layer_batch_output(&self, batch_size: usize, layer: usize) -> Matrix2d<f32> {
        self.layers[layer].new_batch_output(batch_size)
    }

    pub fn new_batch_output(&self, batch_size: usize) -> Matrix2d<f32> {
        self.layers.last().expect("No layers").new_batch_output(batch_size)
    }

    pub fn forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>) {
        self.test_forward_one_impl(0, input, output);
    }

    pub fn test_forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>) {
        self.test_forward_one_impl(0, input, output);
    }

    pub fn train_forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>) {
        self.train_forward_one_impl(0, input, output);
    }

    fn test_forward_one_impl(&self, layer: usize, input: &Vector<f32>, output: &mut Vector<f32>) {
        if layer < self.layers.len() - 1 {
            let mut next_output = self.layers[layer].new_output();
            self.layers[layer].test_forward_one(input, &mut next_output);
            self.test_forward_one_impl(layer + 1, &next_output, output);
        } else {
            self.layers[layer].test_forward_one(input, output);
        }
    }

    fn train_forward_one_impl(&self, layer: usize, input: &Vector<f32>, output: &mut Vector<f32>) {
        if layer < self.layers.len() - 1 {
            let mut next_output = self.layers[layer].new_output();
            self.layers[layer].train_forward_one(input, &mut next_output);
            self.train_forward_one_impl(layer + 1, &next_output, output);
        } else {
            self.layers[layer].train_forward_one(input, output);
        }
    }

    pub fn forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        self.test_forward_batch_impl(0, input, output);
    }

    pub fn test_forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        self.test_forward_batch_impl(0, input, output);
    }

    pub fn train_forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        self.train_forward_batch_impl(0, input, output);
    }

    fn test_forward_batch_impl(&self, layer: usize, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        if layer < self.layers.len() - 1 {
            let mut next_output = self.layers[layer].new_batch_output(input.rows());
            self.layers[layer].test_forward_batch(input, &mut next_output);
            self.test_forward_batch_impl(layer + 1, &next_output, output);
        } else {
            self.layers[layer].test_forward_batch(input, output);
        }
    }

    fn train_forward_batch_impl(&self, layer: usize, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        if layer < self.layers.len() - 1 {
            let mut next_output = self.layers[layer].new_batch_output(input.rows());
            self.layers[layer].train_forward_batch(input, &mut next_output);
            self.train_forward_batch_impl(layer + 1, &next_output, output);
        } else {
            self.layers[layer].train_forward_batch(input, output);
        }
    }

    pub fn forward_batch_layer(&self, layer: usize, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        self.layers[layer].test_forward_batch(input, output);
    }

    pub fn test_forward_batch_layer(&self, layer: usize, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        self.layers[layer].test_forward_batch(input, output);
    }
    pub fn train_forward_batch_layer(&self, layer: usize, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        self.layers[layer].train_forward_batch(input, output);
    }

    pub fn pretty_print(&self) {
        let mut rows = Vec::new();

        let headers = vec!["Index".to_string(), "Layer".to_string(), "Parameters".to_string(), "Output Shape".to_string()];
        rows.push(headers);

        for (i, layer) in self.layers.iter().enumerate() {
            let row = vec![i.to_string(), layer.pretty_name(), layer.parameters().to_string(), layer.output_shape()];
            rows.push(row);
        }

        print_table(rows);
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}
