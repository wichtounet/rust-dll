use std::fs::File;
use std::io::prelude::*;

use etl::vector::Vector;

fn read_header(data: &[u8], pos: usize) -> u32 {
    ((data[pos] as u32) << 24) + ((data[pos + 1] as u32) << 16) + ((data[pos + 2] as u32) << 8) + (data[pos + 3] as u32)
}

fn read_mnist_images_1d(train: bool) -> Vec<Vector<i32>> {
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

    let mut images = Vec::<Vector<i32>>::new();

    for i in 0..count {
        let mut image = Vector::<i32>::new((rows * columns) as usize);

        for c in 0..(rows * columns) as usize {
            image[c] = data[12 + i * rows * columns + c] as i32;
        }

        images.push(image);
    }

    images
}

fn main() {
    println!("Hello, world!");

    let train_images = read_mnist_images_1d(true);
    let test_images = read_mnist_images_1d(false);
}
