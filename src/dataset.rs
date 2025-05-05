use etl::matrix_2d::Matrix2d;

use rand::Rng;

pub trait Dataset {
    fn next_batch(&mut self) -> bool;
    fn last(&self) -> bool;
    fn reset(&mut self);
    fn reset_before_epoch(&mut self);

    fn input_batch(&self) -> &Matrix2d<f32>;
    fn label_batch(&self) -> &Matrix2d<f32>;

    fn batches(&self) -> usize;
    fn index(&self) -> usize;
}

pub struct MemoryDataset {
    input_batches: Vec<Matrix2d<f32>>,
    label_batches: Vec<Matrix2d<f32>>,
    current_batch: usize,
    shuffling: bool,
}

impl MemoryDataset {
    pub fn new(input_batches: Vec<Matrix2d<f32>>, label_batches: Vec<Matrix2d<f32>>) -> Self {
        Self {
            input_batches,
            label_batches,
            current_batch: 0,
            shuffling: false,
        }
    }

    pub fn enable_shuffling(&mut self) {
        self.shuffling = true;
    }
}

impl Dataset for MemoryDataset {
    fn next_batch(&mut self) -> bool {
        self.current_batch += 1;
        self.current_batch < self.batches()
    }

    fn last(&self) -> bool {
        self.current_batch == self.batches() - 1
    }

    fn index(&self) -> usize {
        self.current_batch
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

    fn reset_before_epoch(&mut self) {
        if self.shuffling {
            let mut rng = rand::rng();

            for i in 0..self.batches() - 1 {
                let next = rng.random_range(0..self.batches() - 1);

                self.input_batches.swap(i, next);
                self.label_batches.swap(i, next);
            }
        }

        self.reset();
    }
}
