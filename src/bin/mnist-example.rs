use dll::counters::{dump_counters, dump_counters_pretty};
use dll::dataset::MemoryDataset;
use dll::dense_layer::DenseLayer;
use dll::mnist::*;
use dll::network::Network;
use dll::sgd::Sgd;

fn main() {
    let batch_size = 100;

    let mut train_images = read_mnist_images_1d(true);
    let mut test_images = read_mnist_images_1d(false);

    normalize_images_1d(&mut train_images);
    normalize_images_1d(&mut test_images);

    let train_cat_labels = read_mnist_categorical_labels(true);
    let test_cat_labels = read_mnist_categorical_labels(false);

    let test_batches = images_1d_to_batches(&test_images, batch_size);
    let train_batches = images_1d_to_batches(&train_images, batch_size);

    let test_cat_label_batches = categorical_labels_to_batches(&test_cat_labels, batch_size);
    let train_cat_label_batches = categorical_labels_to_batches(&train_cat_labels, batch_size);

    let mut train_dataset = MemoryDataset::new(train_batches, train_cat_label_batches);
    let mut test_dataset = MemoryDataset::new(test_batches, test_cat_label_batches);

    let mut mlp = Network::new();
    mlp.add_layer(Box::new(DenseLayer::new_sigmoid(28 * 28, 500)));
    mlp.add_layer(Box::new(DenseLayer::new_sigmoid(500, 500)));
    mlp.add_layer(Box::new(DenseLayer::new_stable_softmax(500, 10)));

    mlp.pretty_print();

    let mut trainer = Sgd::new_momentum(&mut mlp, batch_size, false);
    trainer.train(1, &mut train_dataset);

    let (loss, error) = trainer.compute_metrics_dataset(&mut test_dataset).expect("Test metrics should work");
    println!("test: error: {error} loss: {loss}");

    println!("Performance counters");
    dump_counters_pretty();
}
