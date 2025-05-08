use etl::batch_outer_expr::batch_outer;
use etl::batch_softmax_expr::batch_softmax;
use etl::batch_stable_softmax_expr::batch_stable_softmax;
use etl::bias_add_expr::bias_add;
use etl::bias_batch_sum_expr::bias_batch_sum;
use etl::constant::cst;
use etl::etl_expr::EtlExpr;
use etl::matrix_2d::Matrix2d;
use etl::relu_derivative_expr::relu_derivative;
use etl::relu_expr::relu;
use etl::sigmoid_derivative_expr::sigmoid_derivative;
use etl::sigmoid_expr::sigmoid;
use etl::softmax_expr::softmax;
use etl::stable_softmax_expr::stable_softmax;
use etl::transpose_expr::transpose;
use etl::vector::Vector;

use crate::counters::*;
use crate::layer::{Activation, Layer};

pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Matrix2d<f32>,
    biases: Vector<f32>,
    activation: Activation,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut s = Self {
            input_size,
            output_size,
            weights: Matrix2d::<f32>::new_rand_normal(input_size, output_size), // A normal distribution is a decent initialization for weights
            biases: Vector::<f32>::new(output_size),                            // 0 is a good initialization for biases
            activation,
        };

        // Yann Lecun's recommendation for weights initialization
        s.weights /= cst((input_size as f32).sqrt());

        s
    }

    pub fn new_sigmoid(input_size: usize, output_size: usize) -> Self {
        Self::new(input_size, output_size, Activation::Sigmoid)
    }

    pub fn new_relu(input_size: usize, output_size: usize) -> Self {
        Self::new(input_size, output_size, Activation::ReLU)
    }

    pub fn new_softmax(input_size: usize, output_size: usize) -> Self {
        Self::new(input_size, output_size, Activation::Softmax)
    }

    pub fn new_stable_softmax(input_size: usize, output_size: usize) -> Self {
        Self::new(input_size, output_size, Activation::StableSoftmax)
    }
}

impl Layer for DenseLayer {
    fn test_forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>) {
        if self.activation == Activation::Sigmoid {
            *output |= sigmoid(input * &self.weights + &self.biases);
        } else if self.activation == Activation::ReLU {
            *output |= relu(input * &self.weights + &self.biases);
        } else if self.activation == Activation::StableSoftmax {
            *output |= stable_softmax(input * &self.weights + &self.biases);
        } else {
            *output |= softmax(input * &self.weights + &self.biases);
        }
    }

    fn test_forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        let _counter = Counter::new("dense:forward");

        if self.activation == Activation::Sigmoid {
            *output |= sigmoid(bias_add(input * &self.weights, &self.biases));
        } else if self.activation == Activation::ReLU {
            *output |= relu(bias_add(input * &self.weights, &self.biases));
        } else if self.activation == Activation::StableSoftmax {
            *output |= batch_stable_softmax(bias_add(input * &self.weights, &self.biases));
        } else {
            *output |= batch_softmax(bias_add(input * &self.weights, &self.biases));
        }
    }

    fn train_forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>) {
        self.test_forward_one(input, output);
    }

    fn train_forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        self.test_forward_batch(input, output);
    }

    fn new_output(&self) -> Vector<f32> {
        Vector::<f32>::new(self.output_size)
    }

    fn new_batch_output(&self, batch_size: usize) -> Matrix2d<f32> {
        Matrix2d::<f32>::new(batch_size, self.output_size)
    }

    fn adapt_errors(&self, output: &Matrix2d<f32>, errors: &mut Matrix2d<f32>) {
        let _counter = Counter::new("dense:adapt");

        if self.activation == Activation::Sigmoid {
            *errors >>= sigmoid_derivative(output);
        } else if self.activation == Activation::ReLU {
            *errors >>= relu_derivative(output);
        }

        // The derivative of softmax is 1.0
    }

    fn backward_batch(&self, output: &mut Matrix2d<f32>, errors: &Matrix2d<f32>) {
        let _counter = Counter::new("dense:backward");

        *output |= errors * transpose(&self.weights);
    }

    fn new_b_gradients(&self) -> Vector<f32> {
        Vector::<f32>::new(self.output_size)
    }

    fn new_w_gradients(&self) -> Matrix2d<f32> {
        Matrix2d::<f32>::new(self.input_size, self.output_size)
    }

    fn compute_w_gradients(&self, gradients: &mut Matrix2d<f32>, input: &Matrix2d<f32>, errors: &Matrix2d<f32>) {
        let _counter = Counter::new("dense:compute_w");

        *gradients |= batch_outer(input, errors);
    }

    fn compute_b_gradients(&self, gradients: &mut Vector<f32>, _input: &Matrix2d<f32>, errors: &Matrix2d<f32>) {
        let _counter = Counter::new("dense:compute_b");

        *gradients |= bias_batch_sum(errors);
    }

    fn apply_w_gradients(&mut self, gradients: &Matrix2d<f32>) {
        let _counter = Counter::new("dense:apply_w");

        self.weights += gradients;
    }

    fn apply_b_gradients(&mut self, gradients: &Vector<f32>) {
        let _counter = Counter::new("dense:apply_b");

        self.biases += gradients;
    }

    fn pretty_name(&self) -> String {
        format!("Dense ({})", self.activation.to_string())
    }

    fn output_shape(&self) -> String {
        format!("[Bx{}]", self.output_size)
    }

    fn parameters(&self) -> usize {
        self.weights.size()
    }
}
