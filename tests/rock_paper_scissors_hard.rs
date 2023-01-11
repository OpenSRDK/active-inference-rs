use crate::opensrdk_probability::rand::SeedableRng;
use cmaes::DVector;
use opensrdk_kernel_method::*;
use opensrdk_probability::nonparametric::*;
use opensrdk_probability::rand::RngCore;
use opensrdk_probability::rand::{prelude::StdRng, Rng};

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
    history1: Vec<HistoryPage1>,
    history2: Vec<HistoryPage2>,
}

#[derive(Clone, Debug, PartialEq)]
struct HistoryPage1 {
    b1_sigma2: [f64; 2],
    sigma1: [f64; 2],
    previous_b1_sigma2: [f64; 2],
}

#[derive(Clone, Debug, PartialEq)]
struct HistoryPage2 {
    b2_sigma1: [f64; 2],
    sigma2: [f64; 2],
    previous_b2_sigma1: [f64; 2],
}

#[test]
fn test_main() {
    let mut rng = StdRng::from_seed([1; 32]);

    let mut data = Data {
        history1: vec![],
        history2: vec![],
    };
    let mut sigma1 = [0.25, 0.25];
    let mut sigma2 = [0.25, 0.25];
    let mut b1_sigma2 = [0.2, 0.2];
    let mut b2_sigma1 = [0.2, 0.2];
    let mut t = 1usize;
    let mut count: [u32; 2] = [0, 0];
    loop {
        let hand1 = random_hand(&sigma1, &mut rng);
        let hand2 = random_hand(&sigma2, &mut rng);
        result_count(hand1, hand2, &mut count);

        // println!("{:#?}, {:#?}", hand1, hand2);

        let previous_b1_sigma2 = b1_sigma2.clone();
        let previous_b2_sigma1 = b2_sigma1.clone();
        b1_sigma2 = update_belief(t, hand2, &previous_b1_sigma2);
        b2_sigma1 = update_belief(t, hand1, &previous_b2_sigma1);

        if t % 100 == 0 {
            learn2_by_1(&mut data, &b1_sigma2, &sigma1, &previous_b1_sigma2);
            learn1_by_2(&mut data, &b2_sigma1, &sigma2, &previous_b2_sigma1);

            let previous_sigma1 = sigma1.clone();
            let previous_sigma2 = sigma2.clone();
            sigma1 = optimize_policy1(&data, &previous_sigma1, &previous_b1_sigma2);
            sigma2 = optimize_policy2(&data, &previous_sigma2, &previous_b2_sigma1);

            println!("sigma1: {:#?}, b1_sigma2: {:#?}", sigma1, b1_sigma2);
            println!("sigma2: {:#?}, b2_sigma1: {:#?}", sigma2, b2_sigma1);
            println!("[1の勝ち, 2の勝ち] = {:#?}", count);
        }
        t = t + 1;
    }
}

fn random_hand(sigma: &[f64; 2], rng: &mut dyn RngCore) -> Hand {
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

fn result_count(my_hand: Hand, others_hand: Hand, count: &mut [u32; 2]) -> &[u32; 2] {
    match my_hand {
        Hand::Rock => {
            if others_hand == Hand::Paper {
                count[1] += 1
            } else if others_hand == Hand::Scissors {
                count[0] += 1
            }
        }
        Hand::Paper => {
            if others_hand == Hand::Rock {
                count[0] += 1
            } else if others_hand == Hand::Scissors {
                count[1] += 1
            }
        }
        Hand::Scissors => {
            if others_hand == Hand::Rock {
                count[1] += 1
            } else if others_hand == Hand::Paper {
                count[0] += 1
            }
        }
    }
    count
}

fn update_belief(t: usize, others_hand: Hand, others_policy: &[f64; 2]) -> [f64; 2] {
    let mut result = others_policy.clone();
    let t = if t > 10 { 10.0 } else { t as f64 };
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
    data.history1.push(HistoryPage1 {
        b1_sigma2: b1_sigma2.clone(),
        sigma1: sigma1.clone(),
        previous_b1_sigma2: previous_b1_sigma2.clone(),
    });

    if data.history1.len() > 500 {
        data.history1.remove(0);
    }

    // println!("data: {:#?}", data);
}

fn learn1_by_2(
    data: &mut Data,
    b2_sigma1: &[f64; 2],
    sigma2: &[f64; 2],
    previous_b2_sigma1: &[f64; 2],
) {
    data.history2.push(HistoryPage2 {
        b2_sigma1: b2_sigma1.clone(),
        sigma2: sigma2.clone(),
        previous_b2_sigma1: previous_b2_sigma1.clone(),
    });

    if data.history2.len() > 500 {
        data.history2.remove(0);
    }

    // println!("data: {:#?}", data);
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

    let base_params = BaseEllipticalProcessParams::new(kernel, x, theta, sigma).unwrap();

    let params_rock_prop_scissors = base_params.clone().exact(y_rock_prop_scissors).unwrap();
    let result_rock_prop_scissors = params_rock_prop_scissors.gp_predict(&xs).unwrap().mu();

    // println!("r_rock_scissors: {}", result_rock_prop_scissors);

    let params_paper_prop_scissors = base_params.exact(y_paper_prop_scissors).unwrap();
    let result_paper_prop_scissors = params_paper_prop_scissors.gp_predict(&xs).unwrap().mu();

    // println!("r_paper_scissors: {}", result_paper_prop_scissors);

    let scissors = 1.0 / (1.0 / result_rock_prop_scissors + 1.0 / result_paper_prop_scissors + 1.0);
    let rock = scissors / result_rock_prop_scissors;
    let paper = scissors / result_paper_prop_scissors;

    [rock, paper]
}

fn predict2_by_1(data: &Data, sigma1: &[f64; 2], previous_b1_sigma2: &[f64; 2]) -> [f64; 2] {
    let y_rock_prop_scissors = data
        .history1
        .iter()
        .map(|page| page.b1_sigma2)
        .map(|b1_sigma2| (1.0 - b1_sigma2[0] - b1_sigma2[1]) / b1_sigma2[0])
        // .map(|prop| prop.ln())
        .collect::<Vec<f64>>();
    let y_paper_prop_scissors = data
        .history1
        .iter()
        .map(|page| page.b1_sigma2)
        .map(|b1_sigma2| (1.0 - b1_sigma2[0] - b1_sigma2[1]) / b1_sigma2[1])
        // .map(|prop| prop.ln())
        .collect::<Vec<f64>>();

    // println!("his_r_s_2by1: {:#?}", y_rock_prop_scissors);
    // println!("his_p_s_2by1: {:#?}", y_paper_prop_scissors);

    let x = data
        .history1
        .iter()
        .map(|page| (page.sigma1, page.previous_b1_sigma2))
        .map(|(sigma1, previous_b1_sigma2)| [sigma1.to_vec(), previous_b1_sigma2.to_vec()].concat())
        .collect::<Vec<_>>();

    let xs = [sigma1.to_vec(), previous_b1_sigma2.to_vec()].concat();

    predict(&y_rock_prop_scissors, &y_paper_prop_scissors, x, xs)
}

fn predict1_by_2(data: &Data, sigma2: &[f64; 2], previous_b2_sigma1: &[f64; 2]) -> [f64; 2] {
    let y_rock_prop_scissors = data
        .history2
        .iter()
        .map(|page| page.b2_sigma1)
        .map(|b2_sigma1| (1.0 - b2_sigma1[0] - b2_sigma1[1]) / b2_sigma1[0])
        // .map(|prop| prop.ln())
        .collect::<Vec<f64>>();
    let y_paper_prop_scissors = data
        .history2
        .iter()
        .map(|page| page.b2_sigma1)
        .map(|b2_sigma1| (1.0 - b2_sigma1[0] - b2_sigma1[1]) / b2_sigma1[1])
        // .map(|prop| prop.ln())
        .collect::<Vec<f64>>();

    // println!("his_r_s_1by2: {:#?}", y_rock_prop_scissors);
    // println!("his_p_s_1by2: {:#?}", y_paper_prop_scissors);

    let x = data
        .history2
        .iter()
        .map(|page| (page.sigma2, page.previous_b2_sigma1))
        .map(|(sigma2, previous_b2_sigma1)| [sigma2.to_vec(), previous_b2_sigma1.to_vec()].concat())
        .collect::<Vec<_>>();

    let xs = [sigma2.to_vec(), previous_b2_sigma1.to_vec()].concat();

    predict(&y_rock_prop_scissors, &y_paper_prop_scissors, x, xs)
}

fn expected_utility(my_policy: &[f64; 2], others_policy: &[f64; 2]) -> f64 {
    let my_policy = [
        my_policy[0],
        my_policy[1],
        1.0 - my_policy[0] - my_policy[1],
    ];
    let others_policy = [
        others_policy[0],
        others_policy[1],
        1.0 - others_policy[0] - others_policy[1],
    ];

    let utility = 3.0
        * (my_policy[0] * others_policy[2]
            + my_policy[1] * others_policy[0]
            + my_policy[2] * others_policy[1])
        + 1.0
            * (my_policy[0] * others_policy[0]
                + my_policy[1] * others_policy[1]
                + my_policy[2] * others_policy[2])
        + 0.0
            * (my_policy[0] * others_policy[1]
                + my_policy[1] * others_policy[2]
                + my_policy[2] * others_policy[0]);

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
            Hand::Scissors => 1.0 - previous_my_policy[0] - previous_my_policy[1],
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

fn distance(sigma: &[f64; 2], sigma_prime: &[f64; 2]) -> f64 {
    ((sigma[0] - sigma_prime[0]).powi(2) + (sigma[1] - sigma_prime[1]).powi(2)).sqrt()
}

fn optimize_policy1(
    data: &Data,
    previous_sigma1: &[f64; 2],
    previous_b1_sigma2: &[f64; 2],
) -> [f64; 2] {
    let func_to_minimize = |b1_sigma2_prediction: &[f64; 2]| {
        let b1_sigma2_prediction_prime = predict2_by_1(
            data,
            &best_response1(b1_sigma2_prediction, previous_sigma1),
            previous_b1_sigma2,
        );
        distance(b1_sigma2_prediction, &b1_sigma2_prediction_prime)
    };

    let solution = cmaes::fmin(
        |x: &DVector<f64>| func_to_minimize(&[x[0], x[1]]),
        previous_b1_sigma2.to_vec(),
        1.0,
    );
    let b1_sigma2_prediction = [solution.point[0], solution.point[1]];

    let sigma1 = best_response1(&b1_sigma2_prediction, previous_sigma1);

    sigma1
}

fn optimize_policy2(
    data: &Data,
    previous_sigma2: &[f64; 2],
    previous_b2_sigma1: &[f64; 2],
) -> [f64; 2] {
    // let func_to_minimize = |b2_sigma1_prediction: &[f64; 2]| {
    //     let b2_sigma1_prediction_prime = predict1_by_2(
    //         data,
    //         &best_response2(b2_sigma1_prediction, previous_sigma2),
    //         previous_b2_sigma1,
    //     );
    //     distance(b2_sigma1_prediction, &b2_sigma1_prediction_prime)
    // };

    // let solution = cmaes::fmin(
    //     |x: &DVector<f64>| func_to_minimize(&[x[0], x[1]]),
    //     previous_b2_sigma1.to_vec(),
    //     1.0,
    // );
    // let b2_sigma1_prediction = [solution.point[0], solution.point[1]];

    let sigma2 = best_response2(&previous_b2_sigma1, previous_sigma2);

    sigma2
}
