use opensrdk_probability::{
    Distribution, DistributionError, RandomVariable, SampleableDistribution,
};

use crate::Observation;

#[derive(Clone, Debug)]
pub struct KnownObservation<S, Ai, AOthers, Oi, D>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    D: Distribution<Value = Oi, Condition = ((Ai, AOthers), S)>,
{
    distr: D,
}

impl<S, Ai, AOthers, Oi, D> KnownObservation<S, Ai, AOthers, Oi, D>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    D: Distribution<Value = Oi, Condition = ((Ai, AOthers), S)>,
{
    pub fn new(distr: D) -> Self {
        Self { distr }
    }
}

impl<S, Ai, AOthers, Oi, D> Distribution for KnownObservation<S, Ai, AOthers, Oi, D>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    D: Distribution<Value = Oi, Condition = ((Ai, AOthers), S)>,
{
    type Value = Oi;
    type Condition = ((Ai, AOthers), S);

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        self.distr.p_kernel(x, theta)
    }
}

impl<S, Ai, AOthers, Oi, D> SampleableDistribution for KnownObservation<S, Ai, AOthers, Oi, D>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    D: SampleableDistribution<Value = Oi, Condition = ((Ai, AOthers), S)>,
{
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn opensrdk_probability::rand::RngCore,
    ) -> Result<Self::Value, DistributionError> {
        self.distr.sample(theta, rng)
    }
}

impl<S, Ai, AOthers, Oi, D> Observation<S, Ai, AOthers, Oi>
    for KnownObservation<S, Ai, AOthers, Oi, D>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    D: Distribution<Value = Oi, Condition = ((Ai, AOthers), S)>,
{
    fn update(
        &mut self,
        _o_i_next: &Oi,
        _a_i: &Ai,
        _a_others: &AOthers,
        _s_next: &S,
    ) -> Result<(), DistributionError> {
        Ok(()) //sとoを入手することで学習するが、全て既知なので必要なし
    }
}
