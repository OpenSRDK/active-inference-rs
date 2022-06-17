use opensrdk_kernel_method::PositiveDefiniteKernel;
use opensrdk_probability::{
    nonparametric::kernel_matrix, Distribution, DistributionError, RandomVariable,
};

use crate::PolicyOthers;

#[derive(Clone, Debug)]
pub struct NonParametricPolicyOthers<S, AOthers, K>
where
    S: RandomVariable,
    AOthers: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    history: Vec<(S, AOthers)>,
    kernel: K,
    kernel_params: Vec<f64>,
}

impl<S, AOthers, K> NonParametricPolicyOthers<S, AOthers, K>
where
    S: RandomVariable,
    AOthers: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    pub fn new(history: Vec<(S, AOthers)>, kernel: K, kernel_params: Vec<f64>) -> Self {
        Self {
            history,
            kernel,
            kernel_params,
        }
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

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let v = std::iter::once([theta.transform_vec().0, x.transform_vec().0].concat())
            .chain(
                self.history
                    .iter()
                    .map(|e| [e.0.transform_vec().0, e.1.transform_vec().0].concat()),
            )
            .collect::<Vec<_>>();

        let n = self.history.len();
        let kernel_matrix = kernel_matrix(&self.kernel, &self.kernel_params, &v, &v).unwrap();

        let mut sum = 0.0;

        for i in 0..n {
            sum += kernel_matrix[0][i + 1].abs()
                / (kernel_matrix[0][0].sqrt() * kernel_matrix[i + 1][i + 1].sqrt());
        }

        Ok(sum / n as f64)
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn opensrdk_probability::rand::RngCore,
    ) -> Result<Self::Value, DistributionError> {
        todo!()
    }
}

impl<S, AOthers, K> PolicyOthers<S, AOthers> for NonParametricPolicyOthers<S, AOthers, K>
where
    S: RandomVariable,
    AOthers: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    fn update(&mut self, s: &S, a_others: &AOthers) -> Result<(), DistributionError> {
        self.history.push((s.clone(), a_others.clone()));
        Ok(())
    }
}
