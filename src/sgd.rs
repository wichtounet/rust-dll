use etl::abs_expr::abs;
use etl::argmax_expr::argmax;
use etl::constant::cst;
use etl::log_expr::log;
use etl::matrix_2d::Matrix2d;
use etl::min_expr::binary_min;
use etl::reductions::sum;
use etl::vector::Vector;

use crate::network::Network;

pub struct Sgd<'a> {
    network: &'a mut Network,
    outputs: Vec<Option<Matrix2d<f32>>>,
    errors: Vec<Option<Matrix2d<f32>>>,
    w_gradients: Vec<Option<Matrix2d<f32>>>,
    b_gradients: Vec<Option<Vector<f32>>>,
    batch_size: usize,
    learning_rate: f32,
}

impl<'a> Sgd<'a> {
    pub fn new(network: &'a mut Network, batch_size: usize) -> Self {
        let mut trainer = Self {
            network,
            outputs: Vec::new(),
            errors: Vec::new(),
            w_gradients: Vec::new(),
            b_gradients: Vec::new(),
            batch_size,
            learning_rate: 0.1,
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

        // initialization of the gradients
        for layer in 0..layers {
            trainer.w_gradients.push(Some(trainer.network.get_layer(layer).new_w_gradients()));
            trainer.b_gradients.push(Some(trainer.network.get_layer(layer).new_b_gradients()));
        }

        trainer
    }

    fn compute_metrics_batch(&mut self, input_batch: &Matrix2d<f32>, label_batch: &Matrix2d<f32>) -> Option<(f32, f32)> {
        let layers = self.network.layers();
        let last_layer = layers - 1;

        // Forward propagation of the batch

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

        // Compute CCE error and loss

        let last_output = self.outputs[last_layer].take()?;

        let alpha: f32 = -1.0 / (self.batch_size as f32);
        let loss: f32 = alpha * sum(&(log(&last_output) >> label_batch));

        let beta: f32 = 1.0 / (self.batch_size as f32);
        let error: f32 = beta * sum(&(binary_min(abs(argmax(label_batch) - argmax(&last_output)), cst(1.0))));

        self.outputs[last_layer] = Some(last_output);

        Some((loss, error))
    }

    fn compute_metrics_dataset(&mut self, input_batches: &Vec<Matrix2d<f32>>, label_batches: &Vec<Matrix2d<f32>>) -> Option<(f32, f32)> {
        let batches = input_batches.len();

        let mut global_loss: f32 = 0.0;
        let mut global_error: f32 = 0.0;

        for batch in 0..batches {
            let (loss, error) = self.compute_metrics_batch(&input_batches[batch], &label_batches[batch])?;

            global_loss += loss;
            global_error += error;
        }

        Some((global_loss / batches as f32, global_error / batches as f32))
    }

    fn train_batch(&mut self, _epoch: usize, input_batch: &Matrix2d<f32>, label_batch: &Matrix2d<f32>) -> Option<(f32, f32)> {
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

            // With categorical cross entropy, there is no need to multiply by the derivative of
            // the activation function since the terms are canceling out in the derivative of the
            // loss
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

        // Compute the gradients

        for layer in 0..layers {
            let mut w_gradients = self.w_gradients[layer].take()?;
            let mut b_gradients = self.b_gradients[layer].take()?;
            let errors = self.errors[layer].take()?;

            if layer == 0 {
                self.network.get_layer(layer).compute_w_gradients(&mut w_gradients, input_batch, &errors);
                self.network.get_layer(layer).compute_b_gradients(&mut b_gradients, input_batch, &errors);
            } else {
                let input = self.outputs[layer - 1].take()?;

                self.network.get_layer(layer).compute_w_gradients(&mut w_gradients, &input, &errors);
                self.network.get_layer(layer).compute_b_gradients(&mut b_gradients, &input, &errors);

                self.outputs[layer - 1] = Some(input);
            }

            self.errors[layer] = Some(errors);
            self.b_gradients[layer] = Some(b_gradients);
            self.w_gradients[layer] = Some(w_gradients);
        }

        // Here we could apply gradients decay

        // Apply the gradients
        for layer in 0..layers {
            let mut w_gradients = self.w_gradients[layer].take()?;
            let mut b_gradients = self.b_gradients[layer].take()?;

            w_gradients >>= cst(self.learning_rate / (self.batch_size as f32));
            b_gradients >>= cst(self.learning_rate / (self.batch_size as f32));

            self.network.get_layer_mut(layer).apply_w_gradients(&w_gradients);
            self.network.get_layer_mut(layer).apply_b_gradients(&b_gradients);

            self.b_gradients[layer] = Some(b_gradients);
            self.w_gradients[layer] = Some(w_gradients);
        }

        // Compute the category cross entropy loss and error

        self.compute_metrics_batch(input_batch, label_batch)
    }

    fn train_epoch(&mut self, epoch: usize, input_batches: &Vec<Matrix2d<f32>>, label_batches: &Vec<Matrix2d<f32>>) -> Option<(f32, f32)> {
        let batches = input_batches.len();
        let last_batch = batches - 1;

        for i in 0..batches - 1 {
            let (loss, error) = self.train_batch(epoch, &input_batches[i], &label_batches[i])?;
            println!("epoch {epoch} batch {i}/{batches} error: {error} loss: {loss}");
        }

        let (loss, error) = self.train_batch(epoch, &input_batches[last_batch], &label_batches[last_batch])?;
        println!("epoch {epoch} batch {last_batch}/{batches} error: {error} loss: {loss}");
        Some((loss, error))
    }

    pub fn train(&mut self, epochs: usize, input_batches: &Vec<Matrix2d<f32>>, label_batches: &Vec<Matrix2d<f32>>) -> Option<(f32, f32)> {
        for epoch in 1..epochs {
            self.train_epoch(epoch, input_batches, label_batches)?;

            let (loss, error) = self.compute_metrics_dataset(input_batches, label_batches)?;
            println!("epoch {epoch}/{epochs} error: {error} loss: {loss}");
        }

        self.train_epoch(epochs, input_batches, label_batches)?;

        let (loss, error) = self.compute_metrics_dataset(input_batches, label_batches)?;
        println!("epoch {epochs}/{epochs} error: {error} loss: {loss}");

        Some((loss, error))
    }
}
