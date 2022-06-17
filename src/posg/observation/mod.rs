pub mod known;
pub mod nonparametric;

pub use known::*;
pub use nonparametric::*;

use opensrdk_probability::{Distribution, DistributionError, RandomVariable};

pub trait Observation<S, Ai, AOthers, Oi>:
    Distribution<Value = Oi, Condition = ((Ai, AOthers), S)>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
{
    fn update(
        &mut self,
        o_i_next: &Oi,
        a_i: &Ai,
        a_others: &AOthers,
        s_next: &S,
    ) -> Result<(), DistributionError>;
}
