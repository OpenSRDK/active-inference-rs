extern crate blas_src;
extern crate lapack_src;
extern crate opensrdk_linear_algebra;
extern crate opensrdk_probability;
extern crate rayon;

pub mod brain;
pub mod environment;
pub mod nerve;

pub use brain::*;
pub use environment::*;
pub use nerve::*;

#[cfg(test)]
mod tests {
  #[test]
  fn it_works() {
    assert_eq!(2 + 2, 4);
  }
}
