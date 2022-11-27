use crate::opensrdk_probability::rand::SeedableRng;
use opensrdk_active_inference::{KnownObservation, NonParametricPolicyOthers, POSGAgent};
use opensrdk_kernel_method::RBF;
use opensrdk_probability::{
    nonparametric::GeneralizedKernelDensity,
    rand::{prelude::StdRng, Rng},
    InstantDistribution, RandomVariable,
};

extern crate blas_src;
extern crate lapack_src;
extern crate opensrdk_linear_algebra;
extern crate opensrdk_probability;
extern crate rayon;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Hand {
    Rock,
    Paper,
    Scissors,
}

impl RandomVariable for Hand {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (
            vec![
                if *self == Hand::Rock { 1.0 } else { 0.0 },
                if *self == Hand::Paper { 1.0 } else { 0.0 },
                if *self == Hand::Scissors { 1.0 } else { 0.0 },
            ],
            (),
        )
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn restore(
        v: &[f64],
        info: &Self::RestoreInfo,
    ) -> Result<Self, opensrdk_probability::DistributionError> {
        todo!()
    }
}

#[test]
fn test_main() {
    let mut rng = StdRng::from_seed([1; 32]);
    let mut agent = POSGAgent::new(
        (Hand::Rock, Hand::Rock),
        KnownObservation::new(InstantDistribution::new(
            |x: &(Hand, Hand), theta: &((Hand, Hand), (Hand, Hand))| {
                //
                if *x == theta.1 && x.0 == theta.0 .0 && x.1 == theta.0 .1 {
                    Ok(1.0)
                } else {
                    Ok(0.0)
                }
            },
            |theta, rng| todo!(),
        )),
        |a: &Hand, s| match s.0 {
            Hand::Rock => match s.1 {
                Hand::Rock => 0.0,
                Hand::Paper => -1.0,
                Hand::Scissors => 1.0,
            },
            Hand::Paper => match s.1 {
                Hand::Rock => 1.0,
                Hand::Paper => 0.0,
                Hand::Scissors => -1.0,
            },
            Hand::Scissors => match s.1 {
                Hand::Rock => -1.0,
                Hand::Paper => 1.0,
                Hand::Scissors => 0.0,
            },
        },
        InstantDistribution::new(|x, theta| todo!(), |theta, rng| todo!()),
        vec![0.1, 0.2, 0.7],
        NonParametricPolicyOthers::new(vec![], RBF, vec![1.0, 1.0]),
    );

    for i in 0..100 {
        let a_i = agent.sample_action(&mut rng).unwrap();
        let a_others = match rng.gen_range(0..3) {
            0 => Hand::Rock,
            1 => Hand::Paper,
            2 => Hand::Scissors,
            _ => panic!(),
        };

        agent.observe(&a_i, &(a_i, a_others)).unwrap();
    }

    // TODO: confirm the convergence into Mixed Strategy Nash Equilibrium [0.33, 0.33, 0.33]
    //自分のpolicyがこうの時に相手のpolicyがこうという関数をnonparaでいいので関数近似したい
    //相手の最適反応関数を使って、ベルマン方程式的なものを考え、方策を学習していく
    //ベルマン方程式は、事実上使えない（Q学習的なものになる）、world modelかpilcoのような手法で方策を学習していくことになる。
    //相手の最適反応関数を考慮に入れて学習すれば、ナッシュ均衡にいたるのでは（仮説）
}
