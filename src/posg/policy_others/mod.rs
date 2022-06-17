pub mod nonparametric;

pub use nonparametric::*;

use opensrdk_probability::{Distribution, DistributionError, RandomVariable};

pub trait PolicyOthers<S, AOthers>: Distribution<Value = AOthers, Condition = S>
where
    S: RandomVariable,
    AOthers: RandomVariable,
{
    fn update(&mut self, s: &S, a_others: &AOthers) -> Result<(), DistributionError>;
}
