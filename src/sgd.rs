use etl::abs_expr::abs;
use etl::argmax_expr::argmax;
use etl::constant::cst;
use etl::log_expr::log;
use etl::matrix_2d::Matrix2d;
use etl::min_expr::binary_min;
use etl::reductions::sum;
use etl::sqrt_expr::sqrt;
use etl::vector::Vector;
use std::time::Instant;

use crate::counters::*;
use crate::dataset::Dataset;
use crate::network::Network;

#[derive(PartialEq)]
pub enum TrainMethod {
    Sgd,
    Momentum,
    NAdam, // Nesterov Adam
}

pub struct Sgd<'a> {
    network: &'a mut Network,
    outputs: Vec<Option<Matrix2d<f32>>>,
    errors: Vec<Option<Matrix2d<f32>>>,
    w_gradients: Vec<Option<Matrix2d<f32>>>,
    b_gradients: Vec<Option<Vector<f32>>>,
    w_inc: Vec<Matrix2d<f32>>, // For Momentum
    b_inc: Vec<Vector<f32>>,   // For Momentum
    w_m: Vec<Matrix2d<f32>>,   // For Momentum
    b_m: Vec<Vector<f32>>,     // For Momentum
    w_v: Vec<Matrix2d<f32>>,   // For NAdam
    b_v: Vec<Vector<f32>>,     // For NAdam
    w_t: Vec<Matrix2d<f32>>,   // For NAdam
    b_t: Vec<Vector<f32>>,     // For NAdam
    schedule: Vec<f32>,        // For NAdam
    iteration: usize,
    batch_size: usize,
    method: TrainMethod,
    learning_rate: f32,
    momentum: f32,
    adam_beta1: f32,
    adam_beta2: f32,
    nadam_schedule_decay: f32,
    verbose: bool,
}

impl<'a> Sgd<'a> {
    pub fn new_sgd(network: &'a mut Network, batch_size: usize, verbose: bool) -> Self {
        Self::new(network, batch_size, verbose, TrainMethod::Sgd)
    }

    pub fn new_momentum(network: &'a mut Network, batch_size: usize, verbose: bool) -> Self {
        Self::new(network, batch_size, verbose, TrainMethod::Momentum)
    }

    pub fn new_nadam(network: &'a mut Network, batch_size: usize, verbose: bool) -> Self {
        Self::new(network, batch_size, verbose, TrainMethod::NAdam)
    }

    pub fn new(network: &'a mut Network, batch_size: usize, verbose: bool, method: TrainMethod) -> Self {
        let mut trainer = Self {
            network,
            outputs: Vec::new(),
            errors: Vec::new(),
            w_gradients: Vec::new(),
            b_gradients: Vec::new(),
            w_inc: Vec::new(),
            b_inc: Vec::new(),
            w_m: Vec::new(),
            b_m: Vec::new(),
            w_v: Vec::new(),
            b_v: Vec::new(),
            w_t: Vec::new(),
            b_t: Vec::new(),
            schedule: Vec::new(),
            iteration: 1,
            batch_size,
            method,
            learning_rate: 0.1,
            momentum: 0.9,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            nadam_schedule_decay: 0.004,
            verbose,
        };

        // The default learning rate must be changed for NAdam
        if trainer.method == TrainMethod::NAdam {
            trainer.learning_rate = 0.002;
        }

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

            if trainer.method == TrainMethod::Momentum {
                trainer.w_inc.push(trainer.network.get_layer(layer).new_w_gradients());
                trainer.b_inc.push(trainer.network.get_layer(layer).new_b_gradients());
            }

            if trainer.method == TrainMethod::NAdam {
                trainer.w_m.push(trainer.network.get_layer(layer).new_w_gradients());
                trainer.b_m.push(trainer.network.get_layer(layer).new_b_gradients());
                trainer.w_v.push(trainer.network.get_layer(layer).new_w_gradients());
                trainer.b_v.push(trainer.network.get_layer(layer).new_b_gradients());
                trainer.w_t.push(trainer.network.get_layer(layer).new_w_gradients());
                trainer.b_t.push(trainer.network.get_layer(layer).new_b_gradients());
                trainer.schedule.push(1.0);
            }
        }

        trainer
    }

    fn compute_metrics_batch(&mut self, input_batch: &Matrix2d<f32>, label_batch: &Matrix2d<f32>, normalize: bool) -> Option<(f32, f32)> {
        let _counter = Counter::new("sgd::compute_metrics");

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

        let alpha: f32 = if normalize { -1.0 / (self.batch_size as f32) } else { -1.0 };
        let loss: f32 = alpha * sum(&(log(&last_output) >> label_batch));

        let beta: f32 = if normalize { 1.0 / (self.batch_size as f32) } else { 1.0 };
        let error: f32 = beta * sum(&(binary_min(abs(argmax(label_batch) - argmax(&last_output)), cst(1.0))));

        self.outputs[last_layer] = Some(last_output);

        Some((loss, error))
    }

    pub fn compute_metrics_dataset(&mut self, dataset: &mut dyn Dataset) -> Option<(f32, f32)> {
        let batches = dataset.batches();

        let mut global_loss: f32 = 0.0;
        let mut global_error: f32 = 0.0;

        dataset.reset();
        while dataset.next_batch() {
            let (loss, error) = self.compute_metrics_batch(&dataset.input_batch(), &dataset.label_batch(), false)?;

            global_loss += loss;
            global_error += error;
        }

        Some((global_loss / (batches * self.batch_size) as f32, global_error / ((batches * self.batch_size) as f32)))
    }

    fn train_batch(&mut self, _epoch: usize, input_batch: &Matrix2d<f32>, label_batch: &Matrix2d<f32>) -> Option<(f32, f32)> {
        let _counter = Counter::new("sgd:train:batch");

        let layers = self.network.layers();
        let last_layer = layers - 1;

        // Forward propagation of the batch

        // This uses a somewhat idiomatic way of doing things with Rust
        // Since we can't borrow two elements of the same vector immutably and mutably at the same
        // time, we must take ownership of them when we need them and then put them back

        {
            let _counter = Counter::new("sgd: forward");

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
        }

        // Compute the errors of the last layer with categorical cross entropy loss

        {
            let _counter = Counter::new("sgd: errors");

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

        {
            let _counter = Counter::new("sgd: backward");

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
        }

        // Compute the gradients

        {
            let _counter = Counter::new("sgd: compute_gradients");

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
        }

        // Here we could apply gradients decay

        // Apply the gradients
        {
            let _counter = Counter::new("sgd: apply_gradients");

            let eps = self.learning_rate;

            for layer in 0..layers {
                let mut w_gradients = self.w_gradients[layer].take()?;
                let mut b_gradients = self.b_gradients[layer].take()?;

                if self.method == TrainMethod::Sgd {
                    w_gradients >>= cst(eps / (self.batch_size as f32));
                    b_gradients >>= cst(eps / (self.batch_size as f32));

                    self.network.get_layer_mut(layer).apply_w_gradients(&w_gradients);
                    self.network.get_layer_mut(layer).apply_b_gradients(&b_gradients);
                } else if self.method == TrainMethod::Momentum {
                    // Since Rust is pretty limited by its borrow-checker, we cannot really combine
                    // proper expressions with mut and non-mut `c`, so we must split the true operation
                    // in two compound operation
                    // This will have a significant performance cost

                    self.w_inc[layer] >>= cst(self.momentum);
                    self.b_inc[layer] >>= cst(self.momentum);

                    self.w_inc[layer] += cst(eps / (self.batch_size as f32)) >> &w_gradients;
                    self.b_inc[layer] += cst(eps / (self.batch_size as f32)) >> &b_gradients;

                    self.network.get_layer_mut(layer).apply_w_gradients(&self.w_inc[layer]);
                    self.network.get_layer_mut(layer).apply_b_gradients(&self.b_inc[layer]);
                } else if self.method == TrainMethod::NAdam {
                    let w_m = &mut self.w_m[layer];
                    let b_m = &mut self.b_m[layer];

                    let w_v = &mut self.w_v[layer];
                    let b_v = &mut self.b_v[layer];

                    let w_t = &mut self.w_t[layer];
                    let b_t = &mut self.b_t[layer];

                    let beta1 = self.adam_beta1;
                    let beta2 = self.adam_beta2;
                    let schedule_decay = self.nadam_schedule_decay;
                    let e = 1e-8;
                    let t = self.iteration as f32;

                    // Compute the schedule for momentum

                    let momentum_cache_t = beta1 * (1.0 - 0.5 * (0.96_f32.powf(t * schedule_decay)));
                    let momentum_cache_t_1 = beta1 * (1.0 - 0.5 * (0.96_f32.powf((t + 1.0) * schedule_decay)));

                    let m_schedule_new = self.schedule[layer] * momentum_cache_t;
                    let m_schedule_next = self.schedule[layer] * momentum_cache_t * momentum_cache_t_1;

                    // We could probably not update the schedule for the biases, but it should work
                    // fine as well that way
                    self.schedule[layer] = m_schedule_new;

                    // Standard Adam estimations of the first and second order moments
                    // Again, thanks to the borrow checker, we must split the expressions

                    *w_m >>= cst(beta1);
                    *w_m += cst(1.0 - beta1) >> &w_gradients;

                    *w_v >>= cst(beta2);
                    *w_v += cst(1.0 - beta2) >> (&w_gradients >> &w_gradients);

                    *b_m >>= cst(beta1);
                    *b_m += cst(1.0 - beta1) >> &b_gradients;

                    *b_v >>= cst(beta2);
                    *b_v += cst(1.0 - beta2) >> (&b_gradients >> &b_gradients);

                    // Correct the bias towards zero of the first and second moments
                    // For performance and memory, we inline this in the last step

                    // Update the parameters

                    let f1 = 1.0 - momentum_cache_t;
                    let f2 = 1.0 - m_schedule_new;

                    let m1 = eps * (f1 / f2);
                    let m2 = eps * momentum_cache_t_1;

                    // Compute the gradients

                    *w_t |= ((cst(m1) >> &w_gradients) + (cst(m2 / (1.0 - m_schedule_next)) >> &*w_m)) / (sqrt(&*w_v / cst(1.0 - beta2.powf(t))) + cst(e));
                    *b_t |= ((cst(m1) >> &b_gradients) + (cst(m2 / (1.0 - m_schedule_next)) >> &*b_m)) / (sqrt(&*b_v / cst(1.0 - beta2.powf(t))) + cst(e));

                    self.network.get_layer_mut(layer).apply_w_gradients(w_t);
                    self.network.get_layer_mut(layer).apply_b_gradients(b_t);
                }

                self.b_gradients[layer] = Some(b_gradients);
                self.w_gradients[layer] = Some(w_gradients);
            }
        }

        self.iteration += 1;

        // Compute the category cross entropy loss and error

        {
            self.compute_metrics_batch(input_batch, label_batch, true)
        }
    }

    fn train_epoch(&mut self, epoch: usize, dataset: &mut dyn Dataset) -> Option<(f32, f32, u128)> {
        let _counter = Counter::new("sgd:train:epoch");

        let start = Instant::now();

        dataset.reset_before_epoch();

        let mut last_loss = 0.0;
        let mut last_error = 0.0;

        while dataset.next_batch() {
            (last_loss, last_error) = self.train_batch(epoch, dataset.input_batch(), dataset.label_batch())?;
            if self.verbose {
                println!("epoch {epoch} batch {}/{} error: {last_error} loss: {last_loss}", dataset.index(), dataset.batches());
            }
        }

        let duration = start.elapsed();
        Some((last_loss, last_error, duration.as_millis()))
    }

    pub fn train(&mut self, epochs: usize, dataset: &mut dyn Dataset) -> Option<(f32, f32)> {
        let _counter = Counter::new("sgd:train");

        println!("Train the network with \"Stochastic Gradient Descent\"");

        for epoch in 1..epochs {
            let (_epoch_loss, _epoch_error, millis) = self.train_epoch(epoch, dataset)?;

            let (loss, error) = self.compute_metrics_dataset(dataset)?;
            println!("epoch {epoch}/{epochs} error: {error} loss: {loss} time: {millis}ms");
        }

        let (_epoch_loss, _epoch_error, millis) = self.train_epoch(epochs, dataset)?;

        let (loss, error) = self.compute_metrics_dataset(dataset)?;
        println!("epoch {epochs}/{epochs} error: {error} loss: {loss} time: {millis}ms");

        Some((loss, error))
    }
}
