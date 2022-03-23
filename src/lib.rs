#[cfg(test)]
extern crate blas_src;
#[cfg(test)]
extern crate lapack_src;
extern crate opensrdk_linear_algebra;
extern crate opensrdk_probability;
extern crate rayon;

use opensrdk_probability::DistributionError;
use std::error::Error;

pub mod pomdp;

pub use pomdp::*;

#[derive(thiserror::Error, Debug)]
pub enum ActiveInferenceError {
    #[error("Distribution error")]
    DistributionError(DistributionError),
    #[error("Others")]
    Others(Box<dyn Error + Send + Sync>),
}

impl From<Box<dyn Error + Send + Sync>> for ActiveInferenceError {
    fn from(e: Box<dyn Error + Send + Sync>) -> Self {
        ActiveInferenceError::Others(e)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
