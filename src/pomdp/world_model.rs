use crate::POMDP;
use opensrdk_probability::RandomVariable;
use std::marker::PhantomData;

pub struct WorldModel<P, X, A, R, const S: usize>
where
    P: POMDP<X, A>,
    X: RandomVariable,
    A: RandomVariable,
    R: Fn(&X, &A, &X) -> f64,
{
    environment: P,
    reward: R,
    history: Vec<(X, [f64; S], A, f64)>,
    phantom: PhantomData<(X, A)>,
}

impl<P, X, A, R, const S: usize> WorldModel<P, X, A, R, S>
where
    P: POMDP<X, A>,
    X: RandomVariable,
    A: RandomVariable,
    R: Fn(&X, &A, &X) -> f64,
{
    pub fn new(environment: P, reward: R) -> Self {
        Self {
            environment,
            reward,
            history: vec![],
            phantom: PhantomData,
        }
    }

    pub fn infer_state(&self, x: &X) -> [f64; S] {
        // infer s from x
        todo!("{:#?}", x);
    }

    pub fn act(&mut self) {
        let x = self.environment.state().clone();
        let s = self.infer_state(&x);
        let a = todo!();

        let x_next = self.environment.transition(&a);

        let r = (self.reward)(&x, &a, x_next);

        self.learn(x, s, a, r, x_next);
    }

    pub fn learn(&mut self, x: X, s: [f64; S], a: A, r: f64, x_next: &X) {
        self.history.push((x, s, a, r));

        todo!("{:#?}", x_next);
    }
}
