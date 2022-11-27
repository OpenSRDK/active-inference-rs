use crate::Observation;
use opensrdk_kernel_method::PositiveDefiniteKernel;
use opensrdk_probability::{
    nonparametric::GeneralizedKernelDensity, Distribution, DistributionError, RandomVariable,
    SampleableDistribution,
};

#[derive(Clone, Debug)]
pub struct NonParametricObservation<S, Ai, AOthers, Oi, K>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    model: GeneralizedKernelDensity<((Ai, AOthers), S), Oi, K>,
}

impl<S, Ai, AOthers, Oi, K> NonParametricObservation<S, Ai, AOthers, Oi, K>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    pub fn new(history: Vec<(((Ai, AOthers), S), Oi)>, kernel: K, kernel_params: Vec<f64>) -> Self {
        let model = GeneralizedKernelDensity::new(history, kernel, kernel_params);
        Self { model }
    }
}

impl<S, Ai, AOthers, Oi, K> Distribution for NonParametricObservation<S, Ai, AOthers, Oi, K>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    type Value = Oi;
    type Condition = ((Ai, AOthers), S);

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        self.model.p_kernel(x, theta)
    }
}

impl<S, Ai, AOthers, Oi, K> SampleableDistribution
    for NonParametricObservation<S, Ai, AOthers, Oi, K>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn opensrdk_probability::rand::RngCore,
    ) -> Result<Self::Value, DistributionError> {
        self.model.sample(theta, rng)
    }
}

impl<S, Ai, AOthers, Oi, K> Observation<S, Ai, AOthers, Oi>
    for NonParametricObservation<S, Ai, AOthers, Oi, K>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    fn update(
        &mut self,
        o_i_next: &Oi,
        a_i: &Ai,
        a_others: &AOthers,
        s_next: &S,
    ) -> Result<(), DistributionError> {
        self.model.history.push((
            a_i.clone(),
            a_others.clone(),
            s_next.clone(),
            o_i_next.clone(),
        ));
        Ok(())
    }
}
