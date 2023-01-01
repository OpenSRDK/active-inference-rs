use crate::opensrdk_probability::rand::SeedableRng;
use opensrdk_active_inference::{KnownObservation, NonParametricPolicyOthers, POSGAgent};
use opensrdk_kernel_method::*;
use opensrdk_probability::nonparametric::*;
use opensrdk_probability::rand::RngCore;
use opensrdk_probability::{
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

#[derive(Clone, Debug, PartialEq)]
struct Data {
    b1_sigma2_history: Vec<[f64; 2]>,
    sigma1_history: Vec<[f64; 2]>,
    previous_b1_sigma2_history: Vec<[f64; 2]>,
    b2_sigma1_history: Vec<[f64; 2]>,
    sigma2_history: Vec<[f64; 2]>,
    previous_b2_sigma1_history: Vec<[f64; 2]>,
}

#[test]
fn test_main() {
    let mut rng = StdRng::from_seed([1; 32]);

    let mut data = Data {
        b1_sigma2_history: vec![],
        sigma1_history: vec![],
        previous_b1_sigma2_history: vec![],
        b2_sigma1_history: vec![],
        sigma2_history: vec![],
        previous_b2_sigma1_history: vec![],
    };
    let mut sigma1 = [0.25, 0.25];
    let mut sigma2 = [0.25, 0.25];
    let mut b1_sigma2 = [0.2, 0.2];
    let mut b2_sigma1 = [0.2, 0.2];
    let mut t = 0usize;
    loop {
        let hand1 = random_hand(&sigma1, &mut rng);
        let hand2 = random_hand(&sigma2, &mut rng);

        let previous_b1_sigma2 = b1_sigma2;
        let previous_b2_sigma1 = b2_sigma1;
        b1_sigma2 = update_belief(t, hand2, &b1_sigma2);
        b2_sigma1 = update_belief(t, hand1, &b2_sigma1);

        learn2_by_1(&mut data, &b1_sigma2, &sigma1, &previous_b1_sigma2);
        learn1_by_2(&mut data, &b2_sigma1, &sigma2, &previous_b2_sigma1);

        let previous_sigma1 = sigma1;
        let previous_sigma2 = sigma2;
        sigma1 = optimize_policy1(&data, &previous_sigma1);
        sigma2 = optimize_policy2(&data, &previous_sigma2);

        t = t + 1;
    }
}

fn random_hand(sigma: &[f64; 2], rng: &mut RngCore) -> Hand {
    let sigma1_r = sigma[0];
    let sigma1_p = sigma[1];
    let random_value = rng.gen_range(0.0..1.0);

    let a_i = if random_value < sigma1_r {
        Hand::Rock
    } else if random_value < sigma1_r + sigma1_p {
        Hand::Paper
    } else {
        Hand::Scissors
    };
    a_i
}

fn update_belief(t: usize, others_hand: Hand, others_policy: &[f64; 2]) -> [f64; 2] {
    let mut result = others_policy.clone();
    let t = t as f64;
    match others_hand {
        Hand::Rock => {
            result[0] = (t * result[0] + 1.0) / (t + 1.0);
            result[1] = (t * result[1] + 0.0) / (t + 1.0);
        }
        Hand::Paper => {
            result[0] = (t * result[0] + 0.0) / (t + 1.0);
            result[1] = (t * result[1] + 1.0) / (t + 1.0);
        }
        Hand::Scissors => {
            result[0] = (t * result[0] + 0.0) / (t + 1.0);
            result[1] = (t * result[1] + 0.0) / (t + 1.0);
        }
    }
    result
}

fn learn2_by_1(
    data: &mut Data,
    b1_sigma2: &[f64; 2],
    sigma1: &[f64; 2],
    previous_b1_sigma2: &[f64; 2],
) {
    data.b1_sigma2_history.push(b1_sigma2.clone());
    data.sigma1_history.push(sigma1.clone());
    data.previous_b1_sigma2_history.push(previous_b1_sigma2);
}

fn learn1_by_2(
    data: &mut Data,
    b2_sigma1: &[f64; 2],
    sigma2: &[f64; 2],
    previous_b2_sigma1: &[f64; 2],
) {
    data.b2_sigma1_history.push(b2_sigma1.clone());
    data.sigma2_history.push(sigma2.clone());
    data.previous_b2_sigma1_history.push(previous_b2_sigma1);
}

fn predict(
    y_rock_prop_scissors: &[f64],
    y_paper_prop_scissors: &[f64],
    x: Vec<Vec<f64>>,
    xs: Vec<f64>,
) -> [f64; 2] {
    let kernel = RBF + Periodic;
    let theta = vec![1.0; kernel.params_len()];
    let sigma = 1.0;

    let params_rock_prop_scissors =
        BaseEllipticalProcessParams::new(kernel.clone(), x.clone(), theta.clone(), sigma)
            .unwrap()
            .exact(&y_rock_prop_scissors)
            .unwrap();

    let result_rock_prop_scissors = params_rock_prop_scissors
        .gp_predict(&xs)
        .unwrap()
        .mu()
        .exp();

    let params_paper_prop_scissors = BaseEllipticalProcessParams::new(kernel, x, theta, sigma)
        .unwrap()
        .exact(&y_paper_prop_scissors)
        .unwrap();

    let result_paper_prop_scissors = params_paper_prop_scissors
        .gp_predict(&xs)
        .unwrap()
        .mu()
        .exp();

    let scissors = 1.0 / (1.0 / result_rock_prop_scissors + 1.0 / result_paper_prop_scissors + 1.0);
    let rock = scissors / result_rock_prop_scissors;
    let paper = scissors / esult_paper_prop_scissors;

    [rock, paper]
}

fn predict2_by_1(data: &Data, sigma1: &[f64; 2], previous_b1_sigma2: &[f64; 2]) -> [f64; 2] {
    let y_rock_prop_scissors = data
        .b1_sigma2_history
        .iter()
        .map(|b1_sigma2| (1.0 - b1_sigma2[0] - b1_sigma2[1]) / b1_sigma2[0])
        .map(|prop| prop.ln())
        .collect::<Vec<f64>>();
    let y_paper_prop_scissors = data
        .b1_sigma2_history
        .iter()
        .map(|b1_sigma2| (1.0 - b1_sigma2[0] - b1_sigma2[1]) / b1_sigma2[1])
        .map(|prop| prop.ln())
        .collect::<Vec<f64>>();

    let x = data
        .sigma1_history
        .iter()
        .zip(data.previous_b1_sigma2_history.iter())
        .map(|(sigma1, previous_b1_sigma2)| [sigma1.to_vec(), previous_b1_sigma2.to_vec()].concat())
        .collect::<Vec<_>>();

    let xs = [sigma1.to_vec(), previous_b1_sigma2].concat();

    predict(&y_rock_prop_scissors, &y_rock_prop_scissors, x, xs)
}

fn predict1_by_2(data: &Data, sigma2: &[f64; 2], previous_b2_sigma1: &[f64; 2]) -> [f64; 2] {
    let y_rock_prop_scissors = data
        .b2_sigma1_history
        .iter()
        .map(|b2_sigma1| (1.0 - b2_sigma1[0] - b2_sigma1[1]) / b2_sigma1[0])
        .map(|prop| prop.ln())
        .collect::<Vec<f64>>();
    let y_paper_prop_scissors = data
        .b2_sigma1_history
        .iter()
        .map(|b2_sigma1| (1.0 - b2_sigma1[0] - b2_sigma1[1]) / b2_sigma1[1])
        .map(|prop| prop.ln())
        .collect::<Vec<f64>>();

    let x = data
        .sigma2_history
        .iter()
        .zip(data.previous_b2_sigma1_history.iter())
        .map(|(sigma2, previous_b2_sigma1)| [sigma2.to_vec(), previous_b2_sigma1.to_vec()].concat())
        .collect::<Vec<_>>();

    let xs = [sigma2.to_vec(), previous_b2_sigma1.to_vec()].concat();

    predict(&y_rock_prop_scissors, &y_rock_prop_scissors, x, xs)
}

fn expected_utility(my_policy: &[f64; 2], others_policy: &[f64; 2]) -> f64 {
    let sigma1_s = 1 as f64 - my_policy[0] - my_policy[1];
    let next_b1_sigma2_s = 1 as f64 - others_policy[0] - others_policy[1];
    let utility = 3 as f64
        * (my_policy[0] * next_b1_sigma2_s
            + my_policy[1] * others_policy[0]
            + sigma1_s * others_policy[1])
        + 1 as f64
            * (my_policy[0] * others_policy[0]
                + my_policy[1] * others_policy[1]
                + sigma1_s * next_b1_sigma2_s);

    utility
}

fn best_response(others_policy: &[f64; 2], previous_my_policy: &[f64; 2]) -> [f64; 2] {
    // https://tsujimotter.hatenablog.com/entry/the-proof-of-the-existence-of-nash-equilibria
    let potential_improvement = |others_policy: &[f64; 2],
                                 previous_my_policy: &[f64; 2],
                                 hand: Hand| {
        let e = match hand {
            Hand::Rock => [1.0, 0.0],
            Hand::Paper => [0.0, 1.0],
            Hand::Scissors => [0.0, 0.0],
        };
        (expected_utility(&e, others_policy) - expected_utility(previous_my_policy, others_policy))
            .max(0.0)
    };

    let improve = |others_policy: &[f64; 2], previous_my_policy: &[f64; 2], hand: Hand| {
        let numerator = match hand {
            Hand::Rock => previous_my_policy[0],
            Hand::Paper => previous_my_policy[1],
            Hand::Scissors => 1 - previous_my_policy[0] - previous_my_policy[1],
        } + potential_improvement(others_policy, previous_my_policy, hand);

        let denominator = 1.0
            + (potential_improvement(others_policy, previous_my_policy, Hand::Rock)
                + potential_improvement(others_policy, previous_my_policy, Hand::Paper)
                + potential_improvement(others_policy, previous_my_policy, Hand::Scissors));

        numerator / denominator
    };

    [
        improve(others_policy, previous_my_policy, Hand::Rock),
        improve(others_policy, previous_my_policy, Hand::Paper),
    ]
}

fn best_response1(b1_sigma2: &[f64; 2], previous_sigma1: &[f64; 2]) -> [f64; 2] {
    best_response(b1_sigma2, previous_sigma1)
}

fn best_response2(b2_sigma1: &[f64; 2], previous_sigma2: &[f64; 2]) -> [f64; 2] {
    best_response(b2_sigma1, previous_sigma2)
}

fn optimize_policy1(data: &Data, previous_sigma1: &[f64; 2]) -> [f64; 2] {
    // let predicted_2 = predict2_by_1(data, sigma1, b1_sigma2);
    todo!()
}

fn optimize_policy2(data: &Data, previous_sigma2: &[f64; 2]) -> [f64; 2] {
    // let predicted_1 = predict1_by_2(data, sigma2, b2_sigma1);
    todo!()
}
