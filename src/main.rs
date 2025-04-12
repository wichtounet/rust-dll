use network::DenseLayer;
use network::Network;

mod mnist;
mod network;

use crate::mnist::*;

fn main() {
    println!("Hello, world!");

    let train_images = read_mnist_images_1d(true);
    let test_images = read_mnist_images_1d(false);

    println!("Train images: {}", train_images.len());
    println!("Test images: {}", test_images.len());

    let train_labels = read_mnist_labels(true);
    let test_labels = read_mnist_labels(false);

    println!("Train labels: {}", train_labels.len());
    println!("Test labels: {}", test_labels.len());

    let mut mlp = Network::new();
    mlp.add_layer(Box::new(DenseLayer::new(28 * 28, 500)));
    mlp.add_layer(Box::new(DenseLayer::new(500, 500)));
    mlp.add_layer(Box::new(DenseLayer::new_softmax(500, 10)));

    let mut output = mlp.new_output();

    mlp.forward_one(test_images.first().expect("No test images"), &mut output);

    let test_batches = images_1d_to_batches(&test_images, 256);
    let train_batches = images_1d_to_batches(&train_images, 256);

    println!("Train batches: {}", train_batches.len());
    println!("Test batches: {}", test_batches.len());

    let mut batch_output = mlp.new_batch_output(256);

    mlp.forward_batch(train_batches.first().expect("No train batch"), &mut batch_output);
}
