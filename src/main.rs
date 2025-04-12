use std::fs::File;
use std::io::prelude::*;

use etl::bias_add_expr::bias_add;
use etl::etl_expr::EtlExpr;
use etl::matrix_2d::Matrix2d;
use etl::vector::Vector;

fn read_header(data: &[u8], pos: usize) -> u32 {
    ((data[pos] as u32) << 24) + ((data[pos + 1] as u32) << 16) + ((data[pos + 2] as u32) << 8) + (data[pos + 3] as u32)
}

fn read_mnist_images_1d(train: bool) -> Vec<Vector<f32>> {
    let mut data: Vec<u8> = Vec::new();

    let path = if train { "datasets/train-images-idx3-ubyte" } else { "datasets/t10k-images-idx3-ubyte" };

    let mut f = File::open(path).expect("File not found");
    f.read_to_end(&mut data).expect("Cannot read the file");

    let magic = read_header(&data, 0);

    if magic != 0x803 {
        panic!("Invalid magic number in MNIST file");
    }

    let count = read_header(&data, 4) as usize;
    let rows = read_header(&data, 8) as usize;
    let columns = read_header(&data, 12) as usize;

    let mut images = Vec::<Vector<f32>>::new();

    for i in 0..count {
        let mut image = Vector::<f32>::new((rows * columns) as usize);

        for c in 0..(rows * columns) as usize {
            image[c] = data[12 + i * rows * columns + c] as f32;
        }

        images.push(image);
    }

    images
}

fn images_1d_to_batches(images: &Vec<Vector<f32>>, batch_size: usize) -> Vec<Matrix2d<f32>> {
    let mut batches = Vec::<Matrix2d<f32>>::new();

    let end = (images.len() / batch_size) * batch_size;

    for b in (0..end).step_by(batch_size) {
        let mut batch = Matrix2d::<f32>::new(batch_size, images[b].size());

        for i in 0..batch_size {
            for p in 0..images[b + i].size() {
                *batch.at_mut(i, p) = images[b + i].at(p);
            }
        }

        batches.push(batch);
    }

    batches
}

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
