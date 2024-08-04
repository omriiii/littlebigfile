use std::fs::File;
use std::io::prelude::*;
use std::path::Path;


fn testPattern(file: &Vec<u8>, pattern: Vec<u8>)
{
    let mut hits = 0;

    let mut byte_idx = 0;
    let mut pattern_idx = 0;

    while byte_idx < file.len()
    {
        for i in 0..8
        {
            pattern_idx = ((byte_idx * 8) + i) % pattern.len();

            if ((file[byte_idx] >> i & 1) == (pattern[pattern_idx])) { hits += 1; }
        }
        byte_idx += 1;
    }


    print!("{}", ((hits as f64) / ((file.len() as f64) * 8.0)))
}
fn createPattern(v: &Vec<u8>, pattern_length: usize) -> Vec<u8>
{
    if pattern_length < 8 { panic!("Oh god oh no!!"); }

    let mut sums: Vec<u32> = vec![0; pattern_length];

    let mut byte_idx = 0;
    let mut sum_idx = 0;

    while byte_idx < v.len()
    {
        for i in 0..8
        {
            sum_idx = ((byte_idx * 8) + i) % pattern_length;

            if ((v[byte_idx] >> i & 1) != 0) { sums[sum_idx] += 1; }
        }
        byte_idx += 1;
    }

    let bit_location_acc: Vec<f64> = sums.iter().map(|x| ((*x as f64) / ((8 * v.len() / pattern_length) as f64)) ).collect();


    let pattern: Vec<u8> = sums.iter().map(|x| ((*x as f64) / ((8 * v.len() / pattern_length) as f64)).round() as u8 ).collect();

    pattern
}

fn main() -> std::io::Result<()>
{
    // Create a path to the file you want to read
    let path = Path::new("file");

    // Open the file in read-only mode
    let mut file = File::open(path)?;

    // Create a mutable vector to store the file contents
    let mut buffer = Vec::new();

    // Read the entire file into the buffer
    file.read_to_end(&mut buffer)?;

    // Print the length of the buffer
    println!("File size: {} bytes", buffer.len());

    // You can now work with the file contents stored in the `buffer` vector

    let pattern = createPattern(&buffer, 512*4096 + 1);
    testPattern(&buffer, pattern);


    //

    Ok(())
}