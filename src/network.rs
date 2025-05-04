use etl::etl_expr::EtlExpr;
use etl::matrix_2d::Matrix2d;
use etl::vector::Vector;

#[derive(PartialEq)]
pub enum Activation {
    Sigmoid,
    Softmax,
    StableSoftmax,
    ReLU,
}

pub trait Layer {
    fn forward_one(&self, input: &Vector<f32>, output: &mut Vector<f32>);
    fn forward_batch(&self, input: &Matrix2d<f32>, output: &mut Matrix2d<f32>);

    fn adapt_errors(&self, output: &Matrix2d<f32>, errors: &mut Matrix2d<f32>);
    fn backward_batch(&self, output: &mut Matrix2d<f32>, errors: &Matrix2d<f32>);

    fn new_output(&self) -> Vector<f32>;
    fn new_batch_output(&self, batch_size: usize) -> Matrix2d<f32>;

    fn new_b_gradients(&self) -> Vector<f32>;
    fn new_w_gradients(&self) -> Matrix2d<f32>;

    fn compute_w_gradients(&self, gradients: &mut Matrix2d<f32>, input: &Matrix2d<f32>, errors: &Matrix2d<f32>);
    fn compute_b_gradients(&self, gradients: &mut Vector<f32>, input: &Matrix2d<f32>, errors: &Matrix2d<f32>);

    fn apply_w_gradients(&mut self, gradients: &Matrix2d<f32>);
    fn apply_b_gradients(&mut self, gradients: &Vector<f32>);

    fn pretty_name(&self) -> String;
    fn output_shape(&self) -> String;
    fn parameters(&self) -> usize;
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

    pub fn pretty_print(&self) {
        let mut rows = Vec::new();

        let headers = vec!["Index".to_string(), "Layer".to_string(), "Parameters".to_string(), "Output Shape".to_string()];
        rows.push(headers);

        for (i, layer) in self.layers.iter().enumerate() {
            let row = vec![i.to_string(), layer.pretty_name(), layer.parameters().to_string(), layer.output_shape()];
            rows.push(row);
        }

        let mut widths = Vec::<usize>::new();

        for row in &rows {
            for (i, column) in row.iter().enumerate() {
                if widths.len() > i {
                    widths[i] = std::cmp::max(widths[i], column.len());
                } else {
                    widths.push(column.len());
                }
            }
        }

        let max_width = 1 + widths.iter().sum::<usize>() + 3 * widths.len();

        for (i, row) in rows.iter().enumerate() {
            if i == 0 {
                println!("{}", "-".repeat(max_width));
            }

            print!("| ");

            for (i, column) in row.iter().enumerate() {
                print!("{:1$} | ", column, widths[i]);
            }

            println!();

            if i == 0 || i == rows.len() - 1 {
                println!("{}", "-".repeat(max_width));
            }
        }
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}
