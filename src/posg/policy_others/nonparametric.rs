use opensrdk_probability::{
    nonparametric::GeneralizedKernelDensity, opensrdk_kernel_method::PositiveDefiniteKernel,
    Distribution, DistributionError, RandomVariable, SampleableDistribution,
};

use crate::PolicyOthers;

#[derive(Clone, Debug)]
pub struct NonParametricPolicyOthers<S, AOthers, K>
where
    S: RandomVariable,
    AOthers: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    distr: GeneralizedKernelDensity<S, AOthers, K>, //基本的にはカーネル密度推定をしたいが、標本の空間が実数スカラー(wikiみたいなナイーブな例)ではなく任意の集合としたい
                                                    //sとa_othersの関係を学習したい
}

impl<S, AOthers, K> NonParametricPolicyOthers<S, AOthers, K>
where
    S: RandomVariable,
    AOthers: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    pub fn from(distr: GeneralizedKernelDensity<S, AOthers, K>) -> Self {
        Self { distr }
    }
    pub fn new(history: Vec<(S, AOthers)>, kernel: K, kernel_params: Vec<f64>) -> Self {
        let distr = GeneralizedKernelDensity::new(history, kernel, kernel_params);
        Self { distr }
    }
}

impl<S, AOthers, K> Distribution for NonParametricPolicyOthers<S, AOthers, K>
where
    S: RandomVariable,
    AOthers: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    type Value = AOthers;
    type Condition = S;

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        self.distr.p_kernel(x, theta)
    }
}

impl<S, AOthers, K> SampleableDistribution for NonParametricPolicyOthers<S, AOthers, K>
where
    S: RandomVariable,
    AOthers: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn opensrdk_probability::rand::RngCore,
    ) -> Result<Self::Value, DistributionError> {
        self.distr.sample(theta, rng)
    }
}

impl<S, AOthers, K> PolicyOthers<S, AOthers> for NonParametricPolicyOthers<S, AOthers, K>
where
    S: RandomVariable,
    AOthers: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    fn update(&mut self, s: &S, a_others: &AOthers) -> Result<(), DistributionError> {
        self.distr.history.push((s.clone(), a_others.clone()));
        Ok(())
    }
}
