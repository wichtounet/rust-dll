use etl::bias_add_expr::bias_add;
use etl::etl_expr::EtlExpr;
use etl::matrix_2d::Matrix2d;
use etl::vector::Vector;

mod mnist;

use crate::mnist::images_1d_to_batches;
use crate::mnist::read_mnist_images_1d;

pub trait Layer {
    fn forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>);
    fn new_output(&self) -> Vector<f32>;

    fn forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>);
    fn new_batch_output(&self, batch_size: usize) -> Matrix2d<f32>;
}

pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Matrix2d<f32>,
    biases: Vector<f32>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            weights: Matrix2d::<f32>::new(input_size, output_size), // TODO Init Weights
            biases: Vector::<f32>::new(output_size),                // 0 is a good initialization for biases
        }
    }
}

impl Layer for DenseLayer {
    fn forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>) {
        *output |= input * &self.weights + &self.biases;
    }

    fn new_output(&self) -> Vector<f32> {
        Vector::<f32>::new(self.output_size)
    }

    fn forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        *output |= bias_add(input * &self.weights, &self.biases);
    }

    fn new_batch_output(&self, batch_size: usize) -> Matrix2d<f32> {
        Matrix2d::<f32>::new(batch_size, self.output_size)
    }
}

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn new_output(&self) -> Vector<f32> {
        self.layers.last().expect("No layers").new_output()
    }

    pub fn new_batch_output(&self, batch_size: usize) -> Matrix2d<f32> {
        self.layers.last().expect("No layers").new_batch_output(batch_size)
    }

    pub fn forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>) {
        self.forward_one_impl(0, input, output);
    }

    fn forward_one_impl(&self, layer: usize, input: &Vector<f32>, output: &mut Vector<f32>) {
        if layer < self.layers.len() - 1 {
            let mut next_output = self.layers[layer].new_output();
            self.layers[layer].forward_one(input, &mut next_output);
            self.forward_one_impl(layer + 1, &next_output, output);
        } else {
            self.layers[layer].forward_one(input, output);
        }
    }

    pub fn forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        self.forward_batch_impl(0, input, output);
    }

    fn forward_batch_impl(&self, layer: usize, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        if layer < self.layers.len() - 1 {
            let mut next_output = self.layers[layer].new_batch_output(input.rows());
            self.layers[layer].forward_batch(input, &mut next_output);
            self.forward_batch_impl(layer + 1, &next_output, output);
        } else {
            self.layers[layer].forward_batch(input, output);
        }
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

fn main() {
    println!("Hello, world!");

    let train_images = read_mnist_images_1d(true);
    let test_images = read_mnist_images_1d(false);

    println!("Train images: {}", train_images.len());
    println!("Test images: {}", test_images.len());

    let mut mlp = Network::new();
    mlp.add_layer(Box::new(DenseLayer::new(28 * 28, 500)));
    mlp.add_layer(Box::new(DenseLayer::new(500, 500)));
    mlp.add_layer(Box::new(DenseLayer::new(500, 10)));

    let mut output = mlp.new_output();

    mlp.forward_one(test_images.first().expect("No test images"), &mut output);

    let test_batches = images_1d_to_batches(&test_images, 256);
    let train_batches = images_1d_to_batches(&train_images, 256);

    println!("Train batches: {}", train_batches.len());
    println!("Test batches: {}", test_batches.len());

    let mut batch_output = mlp.new_batch_output(256);

    mlp.forward_batch(train_batches.first().expect("No train batch"), &mut batch_output);
}
