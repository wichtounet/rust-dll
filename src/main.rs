use counters::dump_counters;
use network::DenseLayer;
use network::Network;
use sgd::Sgd;

mod counters;
mod mnist;
mod network;
mod sgd;

use crate::mnist::*;

fn main() {
    let batch_size = 100;

    let mut train_images = read_mnist_images_1d(true);
    let mut test_images = read_mnist_images_1d(false);

    println!("Train images: {}", train_images.len());
    println!("Test images: {}", test_images.len());

    normalize_images_1d(&mut train_images);
    normalize_images_1d(&mut test_images);

    let train_labels = read_mnist_labels(true);
    let test_labels = read_mnist_labels(false);

    println!("Train labels: {}", train_labels.len());
    println!("Test labels: {}", test_labels.len());

    let train_cat_labels = read_mnist_categorical_labels(true);
    let test_cat_labels = read_mnist_categorical_labels(false);

    println!("Cat. Train labels: {}", train_cat_labels.len());
    println!("Cat. Test labels: {}", test_cat_labels.len());

    let test_batches = images_1d_to_batches(&test_images, batch_size);
    let train_batches = images_1d_to_batches(&train_images, batch_size);

    println!("Train batches: {}", train_batches.len());
    println!("Test batches: {}", test_batches.len());

    let test_label_batches = labels_to_batches(&test_labels, batch_size);
    let train_label_batches = labels_to_batches(&train_labels, batch_size);

    println!("Train label batches: {}", train_label_batches.len());
    println!("Test label batches: {}", test_label_batches.len());

    let test_cat_label_batches = categorical_labels_to_batches(&test_cat_labels, batch_size);
    let train_cat_label_batches = categorical_labels_to_batches(&train_cat_labels, batch_size);

    println!("Cat. Train label batches: {}", train_cat_label_batches.len());
    println!("Cat. Test label batches: {}", test_cat_label_batches.len());

    println!("Test label 0: {}", test_labels.first().expect("No labels"));
    println!("Test label 0: {}", test_cat_labels.first().expect("No labels"));

    let mut mlp = Network::new();
    mlp.add_layer(Box::new(DenseLayer::new_sigmoid(28 * 28, 500)));
    mlp.add_layer(Box::new(DenseLayer::new_sigmoid(500, 500)));
    mlp.add_layer(Box::new(DenseLayer::new_stable_softmax(500, 10)));

    let mut output = mlp.new_output();

    mlp.forward_one(test_images.first().expect("No test images"), &mut output);

    let mut batch_output = mlp.new_batch_output(batch_size);

    mlp.forward_batch(train_batches.first().expect("No train batch"), &mut batch_output);

    let mut trainer = Sgd::new_momentum(&mut mlp, batch_size, false);
    trainer.train(10, &train_batches, &train_cat_label_batches);

    let (loss, error) = trainer
        .compute_metrics_dataset(&test_batches, &test_cat_label_batches)
        .expect("Test metrics should work");

    println!("test: error: {error} loss: {loss}");

    println!("Performance counters");
    dump_counters();
}
