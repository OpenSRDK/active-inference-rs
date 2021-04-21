use crate::Environment;
use crate::Nerve;
use crate::State;
use opensrdk_probability::*;
use rayon::prelude::*;
use std::error::Error;
use std::fmt::Debug;

#[derive(thiserror::Error, Debug)]
pub enum BrainError {
  #[error("Dimension mismatch")]
  DimensionMismatch,
  #[error("Unknown error")]
  Unknown,
}

pub struct Brain<X, D, const C: usize, const S: usize, const A: usize>
where
  X: State,
  D: Distribution<T = X, U = [f64; C]>,
{
  nerves: [Nerve<X>; S],
  x_distr: D,
  x_params: [f64; C],
  x_approx_params: [f64; C],
}

impl<X, D, const C: usize, const S: usize, const A: usize> Brain<X, D, C, S, A>
where
  X: State,
  D: Distribution<T = X, U = [f64; C]>,
{
  pub fn new(
    nerves: [Nerve<X>; S],
    x_distr: D,
    x_params: [f64; C],
  ) -> Result<Self, Box<dyn Error>> {
    Ok(Self {
      nerves,
      x_distr,
      x_params: x_params.clone(),
      x_approx_params: x_params,
    })
  }

  pub fn nerves(&mut self) -> &[Nerve<X>; S] {
    &self.nerves
  }

  pub fn nerves_mut(&mut self) -> &mut [Nerve<X>; S] {
    &mut self.nerves
  }

  pub fn x_params(&mut self) -> &[f64; C] {
    &self.x_params
  }

  pub fn x_params_mut(&mut self) -> &mut [f64; C] {
    &mut self.x_params
  }

  fn sample_s(&mut self, x: &X) -> [f64; S] {
    let mut s = [0.0; S];
    s.par_iter_mut()
      .zip(self.nerves.par_iter_mut())
      .for_each(|(s, nerve)| *s = nerve.sample_s(x));

    s
  }

  fn bayes_est_x(&mut self) -> Result<&[f64; C], Box<dyn Error>> {
    Ok(&self.x_approx_params)
  }

  fn bayes_opt_a(&mut self) -> Result<[bool; A], Box<dyn Error>> {
    Ok([true; A])
  }

  fn learn_delta_s(
    &mut self,
    delta_s: [f64; S],
    s: [f64; S],
    a: [bool; A],
  ) -> Result<(), Box<dyn Error>> {
    Ok(())
  }

  pub fn act<E>(&mut self, env: &mut E) -> Result<(), Box<dyn Error>>
  where
    E: Environment<X, A>,
  {
    let x = env.state();
    let s = self.sample_s(x);

    self.bayes_est_x()?;

    let a = self.bayes_opt_a()?;
    let x = env.transition(a);
    let mut delta_s = self.sample_s(x);
    delta_s
      .par_iter_mut()
      .zip(s.par_iter())
      .for_each(|(delta_si, si)| *delta_si -= si);

    self.learn_delta_s(delta_s, s, a)?;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  #[test]
  fn it_works() {
    assert_eq!(2 + 2, 4);
  }
}
