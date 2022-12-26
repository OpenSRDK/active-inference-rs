use crate::opensrdk_probability::rand::SeedableRng;
use opensrdk_active_inference::{KnownObservation, NonParametricPolicyOthers, POSGAgent};
use opensrdk_kernel_method::*;
use opensrdk_probability::nonparametric::*;
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
    next_b1_sigma2_history: Vec<[f64; 2]>,
    sigma1_history: Vec<[f64; 2]>,
    b1_sigma2_history: Vec<[f64; 2]>,
    next_b2_sigma1_history: Vec<[f64; 2]>,
    sigma2_history: Vec<[f64; 2]>,
    b2_sigma1_history: Vec<[f64; 2]>,
}

#[test]
fn test_main() {
    let mut data = Data {
        next_b1_sigma2_history: vec![],
        sigma1_history: vec![],
        b1_sigma2_history: vec![],
        next_b2_sigma1_history: vec![],
        sigma2_history: vec![],
        b2_sigma1_history: vec![],
    };
    let mut sigma1 = [0.25, 0.25];
    let mut sigma2 = [0.25, 0.25];
    let mut b1_sigma2 = [0.333333, 0.333333];
    let mut b2_sigma1 = [0.333333, 0.333333];
    let mut t = 0usize;
    loop {
        let hand1 = random_hand(&sigma1);
        let hand2 = random_hand(&sigma2);
        let next_b1_sigma2 = update_belief(t, hand2, &b1_sigma2);
        let next_b2_sigma1 = update_belief(t, hand1, &b2_sigma1);

        learn2_by_1(&mut data, &next_b1_sigma2, &sigma1, &b1_sigma2);
        learn1_by_2(&mut data, &next_b2_sigma1, &sigma2, &b2_sigma1);

        sigma1 = optimize_policy1(&data, &sigma1, &b1_sigma2);
        sigma2 = optimize_policy2(&data, &sigma2, &b2_sigma1);

        b1_sigma2 = next_b1_sigma2;
        b2_sigma1 = next_b2_sigma1;
        t = t + 1;
    }
}

fn random_hand(sigma: &[f64; 2]) -> Hand {
    let sigma1_r = sigma[0];
    let sigma1_p = sigma[1];
    let mut rng = StdRng::from_seed([1; 32]);
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
    next_b1_sigma2: &[f64; 2],
    sigma1: &[f64; 2],
    b1_sigma2: &[f64; 2],
) {
    data.next_b1_sigma2_history.push(next_b1_sigma2.clone());
    todo!()
    //学習データ蓄える
}

fn learn1_by_2(
    data: &mut Data,
    next_b2_sigma1: &[f64; 2],
    sigma2: &[f64; 2],
    b2_sigma1: &[f64; 2],
) {
    todo!()
}

fn predict2_by_1(data: &Data, sigma1: &[f64; 2], b1_sigma2: &[f64; 2]) -> [f64; 2] {
    //ガウス過程回帰
    let y_rock = data
        .next_b1_sigma2_history
        .iter()
        .map(|next_b1_sigma2| next_b1_sigma2[0])
        .collect::<Vec<f64>>();
    let y_paper = data
        .next_b1_sigma2_history
        .iter()
        .map(|next_b1_sigma2| next_b1_sigma2[1])
        .collect::<Vec<f64>>();

    let x = data
        .sigma1_history
        .iter()
        .zip(data.b1_sigma2_history.iter())
        .map(|(sigma1, b1_sigma2)| [sigma1.to_vec(), b1_sigma2.to_vec()].concat())
        .collect::<Vec<_>>();

    let kernel = RBF + Periodic;
    let theta = vec![1.0; kernel.params_len()];
    let sigma = 1.0;

    let xs = [sigma1.to_vec(), b1_sigma2.to_vec()].concat();
    let params_rock =
        BaseEllipticalProcessParams::new(kernel.clone(), x.clone(), theta.clone(), sigma)
            .unwrap()
            .exact(&y_rock)
            .unwrap();

    let np_rock = params_rock.gp_predict(&xs).unwrap();
    let mu_rock = np_rock.mu();
    let sigma_rock = np_rock.sigma();

    let params_paper = BaseEllipticalProcessParams::new(kernel, x, theta, sigma)
        .unwrap()
        .exact(&y_paper)
        .unwrap();

    let np_paper = params_paper.gp_predict(&xs).unwrap();
    let mu_paper = np_paper.mu();
    let sigma_paper = np_paper.sigma();

    [mu_rock, mu_paper]
}

fn predict1_by_2(data: &Data, sigma2: &[f64; 2], b2_sigma1: &[f64; 2]) -> [f64; 2] {
    todo!()
}

fn optimize_policy1(data: &Data, sigma1: &[f64; 2], b1_sigma2: &[f64; 2]) -> [f64; 2] {
    let predicted_2 = predict2_by_1(data, sigma1, b1_sigma2);
    todo!()
    //CMA-ES
}

fn optimize_policy2(data: &Data, sigma2: &[f64; 2], b2_sigma1: &[f64; 2]) -> [f64; 2] {
    let predicted_1 = predict1_by_2(data, sigma2, b2_sigma1);
    todo!()
    //CMA-ES
}
