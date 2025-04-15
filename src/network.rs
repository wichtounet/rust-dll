use etl::bias_add_expr::bias_add;
use etl::etl_expr::EtlExpr;
use etl::matrix_2d::Matrix2d;
use etl::sigmoid_derivative_expr::sigmoid_derivative;
use etl::sigmoid_expr::sigmoid;
use etl::softmax_expr::softmax;
use etl::vector::Vector;

#[derive(PartialEq)]
pub enum Activation {
    Sigmoid,
    Softmax,
}

pub trait Layer {
    fn forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>);
    fn forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>);

    fn adapt_errors(&self, output: &Matrix2d<f32>, errors: &mut Matrix2d<f32>);

    fn new_output(&self) -> Vector<f32>;
    fn new_batch_output(&self, batch_size: usize) -> Matrix2d<f32>;
}

pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Matrix2d<f32>,
    biases: Vector<f32>,
    activation: Activation,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            weights: Matrix2d::<f32>::new_rand_normal(input_size, output_size), // A normal distribution is a decent initialization for weights
            biases: Vector::<f32>::new(output_size),                            // 0 is a good initialization for biases
            activation: Activation::Sigmoid,
        }
    }

    pub fn new_softmax(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            weights: Matrix2d::<f32>::new_rand_normal(input_size, output_size), // A normal distribution is a decent initialization for weights
            biases: Vector::<f32>::new(output_size),                            // 0 is a good initialization for biases
            activation: Activation::Softmax,
        }
    }
}

impl Layer for DenseLayer {
    fn forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>) {
        if self.activation == Activation::Sigmoid {
            *output |= sigmoid(input * &self.weights + &self.biases);
        } else {
            *output |= softmax(input * &self.weights + &self.biases);
        }
    }

    fn forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        if self.activation == Activation::Sigmoid {
            *output |= sigmoid(bias_add(input * &self.weights, &self.biases));
        } else {
            *output |= softmax(bias_add(input * &self.weights, &self.biases));
        }
    }

    fn new_output(&self) -> Vector<f32> {
        Vector::<f32>::new(self.output_size)
    }

    fn new_batch_output(&self, batch_size: usize) -> Matrix2d<f32> {
        Matrix2d::<f32>::new(batch_size, self.output_size)
    }

    fn adapt_errors(&self, output: &Matrix2d<f32>, errors: &mut Matrix2d<f32>) {
        if self.activation == Activation::Sigmoid {
            let errors_copy = Matrix2d::<f32>::new_copy(errors);
            *errors |= sigmoid_derivative(output) >> &errors_copy;
        }

        // THe derivative of softmax is 1.0
    }
}

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn get_layer(&mut self, layer: usize) -> &Box<dyn Layer> {
        &self.layers[layer]
    }

    pub fn get_layer_mut(&mut self, layer: usize) -> &mut Box<dyn Layer> {
        &mut self.layers[layer]
    }

    pub fn new_output(&self) -> Vector<f32> {
        self.layers.last().expect("No layers").new_output()
    }

    pub fn new_layer_batch_output(&self, batch_size: usize, layer: usize) -> Matrix2d<f32> {
        self.layers[layer].new_batch_output(batch_size)
    }

    pub fn new_batch_output(&self, batch_size: usize) -> Matrix2d<f32> {
        self.layers.last().expect("No layers").new_batch_output(batch_size)
    }

    pub fn forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>) {
        self.forward_one_impl(0, input, output);
    }

    fn forward_one_impl(&self, layer: usize, input: &Vector<f32>, output: &mut Vector<f32>) {
        if layer < self.layers.len() - 1 {
            let mut next_output = self.layers[layer].new_output();
            self.layers[layer].forward_one(input, &mut next_output);
            self.forward_one_impl(layer + 1, &next_output, output);
        } else {
            self.layers[layer].forward_one(input, output);
        }
    }

    pub fn forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        self.forward_batch_impl(0, input, output);
    }

    fn forward_batch_impl(&self, layer: usize, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        if layer < self.layers.len() - 1 {
            let mut next_output = self.layers[layer].new_batch_output(input.rows());
            self.layers[layer].forward_batch(input, &mut next_output);
            self.forward_batch_impl(layer + 1, &next_output, output);
        } else {
            self.layers[layer].forward_batch(input, output);
        }
    }

    pub fn forward_batch_layer(&self, layer: usize, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>) {
        self.layers[layer].forward_batch(input, output);
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}
