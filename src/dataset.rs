use etl::matrix_2d::Matrix2d;

pub trait Dataset {
    fn next_batch(&mut self) -> bool;
    fn reset(&mut self);

    fn input_batch(&self) -> &Matrix2d<f32>;
    fn label_batch(&self) -> &Matrix2d<f32>;

    fn batches(&self) -> usize;
}

pub struct MemoryDataset {
    input_batches: Vec<Matrix2d<f32>>,
    label_batches: Vec<Matrix2d<f32>>,
    current_batch: usize,
}

impl MemoryDataset {
    pub fn new(input_batches: Vec<Matrix2d<f32>>, label_batches: Vec<Matrix2d<f32>>) -> Self {
        Self {
            input_batches,
            label_batches,
            current_batch: 0,
        }
    }
}

impl Dataset for MemoryDataset {
    fn next_batch(&mut self) -> bool {
        self.current_batch += 1;
        self.current_batch < self.input_batches.len()
    }

    fn reset(&mut self) {
        self.current_batch = 0;
    }

    fn input_batch(&self) -> &Matrix2d<f32> {
        &self.input_batches[self.current_batch]
    }

    fn label_batch(&self) -> &Matrix2d<f32> {
        &self.label_batches[self.current_batch]
    }

    fn batches(&self) -> usize {
        self.input_batches.len()
    }
}
