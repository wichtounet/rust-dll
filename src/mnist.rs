use std::fs::File;
use std::io::prelude::*;

use etl::constant::cst;
use etl::etl_expr::EtlExpr;
use etl::matrix_2d::Matrix2d;
use etl::reductions::{mean, sum};
use etl::vector::Vector;

fn read_header(data: &[u8], pos: usize) -> u32 {
    ((data[pos] as u32) << 24) + ((data[pos + 1] as u32) << 16) + ((data[pos + 2] as u32) << 8) + (data[pos + 3] as u32)
}

pub fn read_mnist_images_1d(train: bool) -> Vec<Vector<f32>> {
    let mut data: Vec<u8> = Vec::new();

    let path = if train { "datasets/train-images-idx3-ubyte" } else { "datasets/t10k-images-idx3-ubyte" };

    let mut f = File::open(path).expect("File not found");
    f.read_to_end(&mut data).expect("Cannot read the file");

    let magic = read_header(&data, 0);

    if magic != 0x803 {
        panic!("Invalid magic number in MNIST file");
    }

    let count = read_header(&data, 4) as usize;
    let rows = read_header(&data, 8) as usize;
    let columns = read_header(&data, 12) as usize;

    let mut images = Vec::<Vector<f32>>::new();

    for i in 0..count {
        let mut image = Vector::<f32>::new((rows * columns) as usize);

        for c in 0..(rows * columns) as usize {
            image[c] = data[16 + i * rows * columns + c] as f32;
        }

        images.push(image);
    }

    images
}

pub fn read_mnist_labels(train: bool) -> Vec<f32> {
    let mut data: Vec<u8> = Vec::new();

    let path = if train { "datasets/train-labels-idx1-ubyte" } else { "datasets/t10k-labels-idx1-ubyte" };

    let mut f = File::open(path).expect("File not found");
    f.read_to_end(&mut data).expect("Cannot read the file");

    let magic = read_header(&data, 0);

    if magic != 0x801 {
        panic!("Invalid magic number in MNIST file");
    }

    let count = read_header(&data, 4) as usize;

    let mut labels = Vec::<f32>::new();

    for i in 0..count {
        let label = data[8 + i] as f32;
        labels.push(label);
    }

    labels
}

pub fn read_mnist_categorical_labels(train: bool) -> Vec<Vector<f32>> {
    let mut data: Vec<u8> = Vec::new();

    let path = if train { "datasets/train-labels-idx1-ubyte" } else { "datasets/t10k-labels-idx1-ubyte" };

    let mut f = File::open(path).expect("File not found");
    f.read_to_end(&mut data).expect("Cannot read the file");

    let magic = read_header(&data, 0);

    if magic != 0x801 {
        panic!("Invalid magic number in MNIST file");
    }

    let count = read_header(&data, 4) as usize;

    let mut labels = Vec::<Vector<f32>>::new();

    for i in 0..count {
        let mut cat_label = Vector::<f32>::new(10); // This is initialized to zero
        let label = data[8 + i] as usize;
        *cat_label.at_mut(label) = 1.0;
        labels.push(cat_label);
    }

    labels
}

pub fn normalize_images_1d(images: &mut [Vector<f32>]) {
    let size = images.len();

    for i in 0..size {
        let image = &mut images[i];

        // TODO Once rust-etl supports it on f32, use mean/stddev

        // Normalize to a mean of zero
        let m = sum(image) / (size as f32);
        *image -= cst(m);

        let mut s: f32 = 0.0;

        for i in 0..image.size() {
            s += image.at(i) * image.at(i);
        }

        s = (s / (size as f32)).sqrt();

        if s != 0.0 {
            *image /= cst(s);
        }
    }
}

pub fn images_1d_to_batches(images: &[Vector<f32>], batch_size: usize) -> Vec<Matrix2d<f32>> {
    let mut batches = Vec::<Matrix2d<f32>>::new();

    let end = (images.len() / batch_size) * batch_size;

    for b in (0..end).step_by(batch_size) {
        let mut batch = Matrix2d::<f32>::new(batch_size, images[b].size());

        for i in 0..batch_size {
            for p in 0..images[b + i].size() {
                *batch.at_mut(i, p) = images[b + i].at(p);
            }
        }

        batches.push(batch);
    }

    batches
}

pub fn labels_to_batches(labels: &[f32], batch_size: usize) -> Vec<Vector<f32>> {
    let mut batches = Vec::<Vector<f32>>::new();

    let end = (labels.len() / batch_size) * batch_size;

    for b in (0..end).step_by(batch_size) {
        let mut batch = Vector::<f32>::new(batch_size);

        for i in 0..batch_size {
            *batch.at_mut(i) = labels[b + i];
        }

        batches.push(batch);
    }

    batches
}

pub fn categorical_labels_to_batches(labels: &[Vector<f32>], batch_size: usize) -> Vec<Matrix2d<f32>> {
    let mut batches = Vec::<Matrix2d<f32>>::new();

    let end = (labels.len() / batch_size) * batch_size;

    for b in (0..end).step_by(batch_size) {
        let mut batch = Matrix2d::<f32>::new(batch_size, 10);

        for i in 0..batch_size {
            for l in 0..10 {
                *batch.at_mut(i, l) = labels[b + i].at(l);
            }
        }

        batches.push(batch);
    }

    batches
}
