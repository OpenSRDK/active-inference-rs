use crate::{NonParametricPolicyOthers, Observation, PolicyOthers};
use opensrdk_kernel_method::RBF;
use opensrdk_probability::{
    rand::{distributions::Uniform, rngs::StdRng, Rng, RngCore, SeedableRng},
    stein::SteinVariational,
    ConditionableDistribution, ContinuousSamplesDistribution, DiscreteUniform, Distribution,
    DistributionError, InstantDistribution, RandomVariable, SampleableDistribution,
};
use std::{clone, marker::PhantomData};

pub struct POSGAgent<S, Ai, AOthers, Oi, Obs, Ri, Pii, Thetai, PiOthers>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    Obs: Observation<S, Ai, AOthers, Oi> + SampleableDistribution,
    Ri: Fn(&Ai, &S) -> f64,
    Pii: Distribution<Value = Ai, Condition = (S, Thetai)>,
    Thetai: RandomVariable,
    PiOthers: PolicyOthers<S, AOthers> + SampleableDistribution,
{
    inferred_state: ContinuousSamplesDistribution<S>,
    observation: Obs,
    reward: Ri,
    policy: Pii,
    theta: Thetai,
    policy_others: PiOthers,
    phantom: PhantomData<(S, Ai, AOthers, Oi)>,
}

impl<S, Ai, AOthers, Oi, Obs, Ri, Pii, Thetai, PiOthers>
    POSGAgent<S, Ai, AOthers, Oi, Obs, Ri, Pii, Thetai, PiOthers>
where
    S: RandomVariable + std::cmp::PartialEq,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    Obs: Observation<S, Ai, AOthers, Oi> + SampleableDistribution,
    Ri: Fn(&Ai, &S) -> f64 + std::marker::Sync,
    Pii: SampleableDistribution<Value = Ai, Condition = (S, Thetai)>,
    Thetai: RandomVariable,
    PiOthers: PolicyOthers<S, AOthers> + SampleableDistribution,
{
    pub fn new(
        inferred_state: ContinuousSamplesDistribution<S>,
        observation: Obs,
        reward: Ri,
        policy: Pii,
        theta: Thetai,
        policy_others: PiOthers,
    ) -> Self {
        Self {
            inferred_state,
            observation,
            reward,
            policy,
            theta,
            policy_others,
            phantom: PhantomData,
        }
    }

    pub fn sample_action(&self, rng: &mut dyn RngCore) -> Result<Ai, DistributionError> {
        self.policy.sample(
            &(self.inferred_state.mean().unwrap(), self.theta.clone()),
            rng,
        )
    }

    pub fn observe(&mut self, a_i: &Ai, o_i_next: &Oi) -> Result<f64, DistributionError> {
        let previous_state_sample_distr = &self.inferred_state;
        let previous_state = &previous_state_sample_distr.mean().unwrap();
        let mut rng = StdRng::from_seed([1; 32]);
        //let (inferred_s_next, inferred_a_others): (S, AOthers) =
        //    todo!("{:#?}{:#?}{:#?}", self.observation, a_i, o_i_next); //(
        //     previous_state.clone(),
        //     self.policy_others.sample(pous_state, &mut rng).unwrap(),
        // );
        //self.observation, a_i, o_i_nextをつかって、inferred_s_next, inferred_a_othersを推定せよ
        //observationとpolicy_othersのトレイトを使って推定する
        //スタイン
        let value = vec![o_i_next];
        let likelihood = InstantDistribution::new(
            |o_i_next, &((_a, inferred_a_others), inferred_s_next)| {
                self.observation.p_kernel(
                    o_i_next,
                    &((a_i.clone(), inferred_a_others), inferred_s_next),
                )
            },
            |&((_a, inferred_a_others), inferred_s_next), rng| {
                self.observation
                    .sample(&((a_i.clone(), inferred_a_others), inferred_s_next), rng)
            },
        );
        let prior_lhs = InstantDistribution::new(
            |inferred_a_others, _s| {
                self.policy_others
                    .p_kernel(inferred_a_others, previous_state)
            },
            |_s, rng| self.policy_others.sample(previous_state, rng),
        );
        let prior_rhs = previous_state_sample_distr.clone();
        let prior = prior_lhs * prior_rhs;
        let kernel = RBF;
        let kernel_params = [0.5, 0.5];
        let samples_orig = (0..10)
            .into_iter()
            .map(|v| {
                let mut rng3 = StdRng::from_seed([v; 32]);
                let theta_0 = rng3.gen_range(-5.0..=5.0);
                let mut rng4 = StdRng::from_seed([v * 2; 32]);
                let theta_1 = rng4.gen_range(-5.0..=5.0);
                vec![theta_0, theta_1]
            })
            .collect::<Vec<Vec<f64>>>();
        let samples_theta = &mut ContinuousSamplesDistribution::new(samples_orig);

        let mut stein = SteinVariational::new(
            value,
            &likelihood,
            &prior,
            &kernel,
            &kernel_params,
            samples_theta,
        );

        let reward = (self.reward)(a_i, &inferred_s_next);

        self.observation
            .update(o_i_next, a_i, &inferred_a_others, &inferred_s_next)?;
        self.policy_others
            .update(previous_state, &inferred_a_others)?;

        self.inferred_state = inferred_s_next;

        Ok(reward)
    }
}
