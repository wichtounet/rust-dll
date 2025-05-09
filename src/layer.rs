use etl::matrix_2d::Matrix2d;
use etl::vector::Vector;

#[derive(PartialEq)]
pub enum Activation {
    Sigmoid,
    Softmax,
    StableSoftmax,
    ReLU,
}

impl Activation {
    pub fn to_string(&self) -> &str {
        match self {
            Activation::Sigmoid => "Sigmoid",
            Activation::Softmax => "Softmax",
            Activation::StableSoftmax => "Softmax (stable)",
            Activation::ReLU => "ReLU",
        }
    }
}

pub trait Layer {
    // Test forward (inference)
    fn test_forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>);
    fn test_forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>);

    // Train forward (inference)
    fn train_forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>);
    fn train_forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>);

    fn adapt_errors(&self, output: &Matrix2d<f32>, errors: &mut Matrix2d<f32>);
    fn backward_batch(&self, output: &mut Matrix2d<f32>, errors: &Matrix2d<f32>);

    fn new_output(&self) -> Vector<f32>;
    fn new_batch_output(&self, batch_size: usize) -> Matrix2d<f32>;

    fn pretty_name(&self) -> String;
    fn output_shape(&self) -> String;
    fn parameters(&self) -> usize;

    /* Only neural layers need to implement gradients function */

    fn new_b_gradients(&self) -> Vector<f32> {
        assert_eq!(self.parameters(), 0);
        panic!("This layer does not have weights");
    }

    fn new_w_gradients(&self) -> Matrix2d<f32> {
        assert_eq!(self.parameters(), 0);
        panic!("This layer does not have weights");
    }

    fn compute_w_gradients(&self, _gradients: &mut Matrix2d<f32>, _input: &Matrix2d<f32>, _errors: &Matrix2d<f32>) {
        assert_eq!(self.parameters(), 0);
        panic!("This layer does not have weights");
    }

    fn compute_b_gradients(&self, _gradients: &mut Vector<f32>, _input: &Matrix2d<f32>, _errors: &Matrix2d<f32>) {
        assert_eq!(self.parameters(), 0);
        panic!("This layer does not have weights");
    }

    fn apply_w_gradients(&mut self, _gradients: &Matrix2d<f32>) {
        assert_eq!(self.parameters(), 0);
        panic!("This layer does not have weights");
    }

    fn apply_b_gradients(&mut self, _gradients: &Vector<f32>) {
        assert_eq!(self.parameters(), 0);
        panic!("This layer does not have weights");
    }
}
