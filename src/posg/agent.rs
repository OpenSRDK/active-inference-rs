use crate::{Observation, PolicyOthers};
use opensrdk_probability::{
    rand::{rngs::StdRng, RngCore, SeedableRng},
    Distribution, DistributionError, RandomVariable,
};
use std::marker::PhantomData;

pub struct POSGAgent<S, Ai, AOthers, Oi, Obs, Ri, Pii, Thetai, PiOthers>
where
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    Obs: Observation<S, Ai, AOthers, Oi>,
    Ri: Fn(&Ai, &S) -> f64,
    Pii: Distribution<Value = Ai, Condition = (S, Thetai)>,
    Thetai: RandomVariable,
    PiOthers: PolicyOthers<S, AOthers>,
{
    inferred_state: S,
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
    S: RandomVariable,
    Ai: RandomVariable,
    AOthers: RandomVariable,
    Oi: RandomVariable,
    Obs: Observation<S, Ai, AOthers, Oi>,
    Ri: Fn(&Ai, &S) -> f64,
    Pii: Distribution<Value = Ai, Condition = (S, Thetai)>,
    Thetai: RandomVariable,
    PiOthers: PolicyOthers<S, AOthers>,
{
    pub fn new(
        inferred_state: S,
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
        self.policy
            .sample(&(self.inferred_state.clone(), self.theta.clone()), rng)
    }

    pub fn observe(&mut self, a_i: &Ai, o_i_next: &Oi) -> Result<f64, DistributionError> {
        let previous_state = &self.inferred_state;
        let mut rng = StdRng::from_seed([1; 32]);
        let (inferred_s_next, inferred_a_others): (S, AOthers) = //(
        //     previous_state.clone(),
        //     self.policy_others.sample(previous_state, &mut rng).unwrap(),
        // );
        todo!("{:#?}{:#?}{:#?}", self.observation, a_i, o_i_next);

        let reward = (self.reward)(a_i, &inferred_s_next);

        self.observation
            .update(o_i_next, a_i, &inferred_a_others, &inferred_s_next)?;
        self.policy_others
            .update(previous_state, &inferred_a_others)?;

        self.inferred_state = inferred_s_next;

        Ok(reward)
    }
}
