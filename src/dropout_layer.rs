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

    fn adapt_errors(&self, _output: &Matrix2d<f32>, _errors: &mut Matrix2d<f32>) {
        // NOP
    }

    fn backward_batch(&self, output: &mut Matrix2d<f32>, errors: &Matrix2d<f32>) {
        let _counter = Counter::new("dropout:backward");
        *output |= errors;
    }

    fn pretty_name(&self) -> String {
        format!("Dropout ({}%)", 100_f32 * self.probability)
    }

    fn parameters(&self) -> usize {
        0
    }

    fn reshapes(&self) -> bool {
        false
    }
}
