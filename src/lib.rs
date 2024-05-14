//! This crate provides an elegant solution for integrating Rayon's parallel processing
//! power with the traditional sequential iterator pattern in Rust.

use std::sync::mpsc::{self, IntoIter};

use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Transforms a Rayon parallel iterator into a sequentially processed iterator.
///
/// This function enables the ergonomic bridging between Rayon's parallel processing capabilities
/// and Rust's sequential iterator paradigm. It achieves this by accepting a Rayon parallel iterator
/// and a function that can work with the items as a standard Rust iterator, effectively allowing
/// parallel computation results to be consumed in a sequential manner.
///
/// The `bound` parameter specifies the buffer size for the channel used to bridge the parallel and
/// sequential computations, allowing some degree of concurrency control.
///
/// # Examples
///
/// ```
/// use rayon_par_bridge::par_bridge;
/// use rayon::prelude::*;
///
/// let data = (0u32..100).collect::<Vec<_>>();
/// let parallel_pipeline = data.into_par_iter().map(|num| num * 2);
///
/// // Use `par_bridge` to consume the parallel pipeline results sequentially
/// let mut result: Vec<_> = par_bridge(5, parallel_pipeline, |seq_iter| seq_iter.collect());
///
/// assert_eq!(result.len(), 100);
///
/// // Numbers can be out-of-order
/// result.sort_unstable();
/// assert_eq!(result[0], 0);
/// assert_eq!(result[1], 2);
/// ```
///
/// # Parameters
/// - `bound`: The size of the internal buffer used to transition items from the parallel
/// pipeline to the sequential iterator. Larger values allow more parallel processing but
/// increase memory usage.
/// - `iter`: The Rayon parallel iterator to be consumed.
/// - `f`: A function that takes a sequential iterator (`RayonIntoIter`) over the parallel
/// iterator's items, enabling sequential processing or collection of the results.
pub fn par_bridge<I, F, R>(bound: usize, iter: I, f: F) -> R
where
    I: IntoParallelIterator + Send,
    F: FnOnce(RayonIntoIter<I::Item>) -> R,
{
    std::thread::scope(|s| {
        let (send, recv) = mpsc::sync_channel(bound);
        s.spawn(move || iter.into_par_iter().try_for_each(|x| send.send(x).ok()));
        f(RayonIntoIter(recv.into_iter()))
    })
}

/// An `Iterator` over the elements returned by a parallel rayon pipeline.
pub struct RayonIntoIter<T>(IntoIter<T>);

impl<T> Iterator for RayonIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[cfg(test)]
mod tests {
    use rayon::prelude::*;

    use super::*;

    #[test]
    fn single_thread() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            let data = (0u32..100).collect::<Vec<_>>();
            let parallel_pipeline = data.into_par_iter().map(|num| num * 2);
            let mut result: Vec<_> =
                par_bridge(5, parallel_pipeline, |seq_iter| seq_iter.collect());

            assert_eq!(result.len(), 100);

            result.sort_unstable();
            assert_eq!(result[0], 0);
            assert_eq!(result[1], 2);
        });
    }
}
