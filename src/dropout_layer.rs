use etl::inv_dropout_mask::inv_dropout_mask;
use etl::matrix_2d::Matrix2d;
use etl::vector::Vector;

use crate::counters::*;
use crate::layer::Layer;

pub struct DropoutLayer {
    probability: f32,
}

impl DropoutLayer {
    pub fn new(probability: f32) -> Self {
        Self { probability }
    }
}

impl Layer for DropoutLayer {
    fn test_forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>) {
        *output |= input;
    }

    fn test_forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        let _counter = Counter::new("dropout:forward:test");

        *output |= input;
    }

    fn train_forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>) {
        *output |= inv_dropout_mask(self.probability) >> input;
    }

    fn train_forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        let _counter = Counter::new("dropout:forward:train");

        *output |= inv_dropout_mask(self.probability) >> input;
    }

    fn new_output(&self) -> Vector<f32> {
        Vector::<f32>::new(1)
    }

    fn new_batch_output(&self, batch_size: usize) -> Matrix2d<f32> {
        Matrix2d::<f32>::new(batch_size, 1)
    }

    fn adapt_errors(&self, _output: &Matrix2d<f32>, _errors: &mut Matrix2d<f32>) {
        // NOP
    }

    fn backward_batch(&self, output: &mut Matrix2d<f32>, errors: &Matrix2d<f32>) {
        let _counter = Counter::new("dropout:backward");
        *output |= errors;
    }

    fn new_b_gradients(&self) -> Vector<f32> {
        Vector::<f32>::new(1)
    }

    fn new_w_gradients(&self) -> Matrix2d<f32> {
        Matrix2d::<f32>::new(1, 1)
    }

    fn compute_w_gradients(&self, _gradients: &mut Matrix2d<f32>, input_: &Matrix2d<f32>, _errors: &Matrix2d<f32>) {}

    fn compute_b_gradients(&self, _gradients: &mut Vector<f32>, _input: &Matrix2d<f32>, _errors: &Matrix2d<f32>) {}

    fn apply_w_gradients(&mut self, _gradients: &Matrix2d<f32>) {}

    fn apply_b_gradients(&mut self, _gradients: &Vector<f32>) {}

    fn pretty_name(&self) -> String {
        format!("Dropout ({}%)", self.probability)
    }

    fn output_shape(&self) -> String {
        format!("[Bx{}]", 1)
    }

    fn parameters(&self) -> usize {
        0
    }
}
