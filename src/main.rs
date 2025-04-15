use etl::etl_expr::EtlExpr;
use etl::matrix_2d::Matrix2d;
use network::DenseLayer;
use network::Network;

mod mnist;
mod network;

use crate::mnist::*;

struct Sgd<'a> {
    network: &'a mut Network,
    outputs: Vec<Option<Matrix2d<f32>>>,
    errors: Vec<Option<Matrix2d<f32>>>,
    w_gradients: Vec<Option<Matrix2d<f32>>>,
    b_gradients: Vec<Option<Matrix2d<f32>>>,
}

impl<'a> Sgd<'a> {
    fn new(network: &'a mut Network, batch_size: usize) -> Self {
        let mut trainer = Self {
            network,
            outputs: Vec::new(),
            errors: Vec::new(),
            w_gradients: Vec::new(),
            b_gradients: Vec::new(),
        };

        let layers = trainer.network.layers();

        // initialization of the outputs
        for layer in 0..layers {
            trainer.outputs.push(Some(trainer.network.get_layer(layer).new_batch_output(batch_size)));
        }

        // initialization of the errors
        for layer in 0..layers {
            trainer.errors.push(Some(trainer.network.get_layer(layer).new_batch_output(batch_size)));
        }

        trainer
    }

    fn train_batch(&mut self, epoch: usize, input_batch: &Matrix2d<f32>, label_batch: &Matrix2d<f32>) -> Option<(f32, f32)> {
        let layers = self.network.layers();
        let last_layer = layers - 1;

        // Forward propagation of the batch

        // This uses a somewhat idiomatic way of doing things with Rust
        // Since we can't borrow two elements of the same vector immutably and mutably at the same
        // time, we must take ownership of them when we need them and then put them back

        for layer in 0..layers {
            let mut output = self.outputs[layer].take()?;

            if layer == 0 {
                self.network.get_layer(layer).forward_batch(input_batch, &mut output);
            } else {
                let input = self.outputs[layer - 1].take()?;
                self.network.get_layer(layer).forward_batch(&input, &mut output);
                self.outputs[layer - 1] = Some(input);
            }

            self.outputs[layer] = Some(output);
        }

        // Compute the errors of the last layer with categorical cross entropy loss

        {
            let mut last_errors = self.errors[last_layer].take()?;
            let last_output = self.outputs[last_layer].take()?;

            last_errors |= label_batch - &last_output;

            self.outputs[last_layer] = Some(last_output);
            self.errors[last_layer] = Some(last_errors);
        }

        // Backward propagation of the errors

        for layer in (0..layers).rev() {
            let mut errors = self.errors[layer].take()?;
            let outputs = self.outputs[layer].take()?;

            // All layers but the last need to adapt errors for the derivative
            if layer != last_layer {
                self.network.get_layer(layer).adapt_errors(&outputs, &mut errors);
            }

            // All layers but the first need back propagation
            if layer > 0 {
                let mut back_errors = self.errors[layer - 1].take()?;
                self.network.get_layer(layer).backward_batch(&mut back_errors, &errors);
                self.errors[layer - 1] = Some(back_errors);
            }

            self.outputs[layer] = Some(outputs);
            self.errors[layer] = Some(errors);
        }

        // TODO Compute errror and loss
        Some((0.0, 0.0))
    }
}

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

    let train_cat_labels = read_mnist_categorical_labels(true);
    let test_cat_labels = read_mnist_categorical_labels(false);

    println!("Cat. Train labels: {}", train_cat_labels.len());
    println!("Cat. Test labels: {}", test_cat_labels.len());

    let test_batches = images_1d_to_batches(&test_images, 256);
    let train_batches = images_1d_to_batches(&train_images, 256);

    println!("Train batches: {}", train_batches.len());
    println!("Test batches: {}", test_batches.len());

    let test_label_batches = labels_to_batches(&test_labels, 256);
    let train_label_batches = labels_to_batches(&train_labels, 256);

    println!("Train label batches: {}", train_label_batches.len());
    println!("Test label batches: {}", test_label_batches.len());

    let test_cat_label_batches = categorical_labels_to_batches(&test_cat_labels, 256);
    let train_cat_label_batches = categorical_labels_to_batches(&train_cat_labels, 256);

    println!("Cat. Train label batches: {}", train_cat_label_batches.len());
    println!("Cat. Test label batches: {}", test_cat_label_batches.len());

    println!("Test label 0: {}", test_labels.first().expect("No labels"));
    println!("Test label 0: {}", test_cat_labels.first().expect("No labels"));

    let mut mlp = Network::new();
    mlp.add_layer(Box::new(DenseLayer::new(28 * 28, 500)));
    mlp.add_layer(Box::new(DenseLayer::new(500, 500)));
    mlp.add_layer(Box::new(DenseLayer::new_softmax(500, 10)));

    let mut output = mlp.new_output();

    mlp.forward_one(test_images.first().expect("No test images"), &mut output);

    let mut batch_output = mlp.new_batch_output(256);

    mlp.forward_batch(train_batches.first().expect("No train batch"), &mut batch_output);

    let mut trainer = Sgd::new(&mut mlp, 256);
    match trainer.train_batch(0, train_batches.first().expect("No train batch"), train_cat_label_batches.first().expect("No train batches")) {
        Some((error, loss)) => println!("error: {error} loss: {loss}"),
        None => println!("Something went wrong during training"),
    }
}
