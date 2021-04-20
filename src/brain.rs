use crate::Environment;
use crate::Nerve;
use crate::State;
use opensrdk_probability::*;
use rayon::prelude::*;
use std::error::Error;
use std::fmt::Debug;
use std::mem::transmute;

#[derive(thiserror::Error, Debug)]
pub enum BrainError {
  #[error("Dimension mismatch")]
  DimensionMismatch,
  #[error("Unknown error")]
  Unknown,
}

pub struct Brain<X, D, const U: usize, const S: usize, const A: usize>
where
  X: State,
  D: Distribution<T = X, U = [f64; U]>,
{
  nerves: [Nerve<X>; S],
  x_distr: D,
  x_params: [f64; U],
  x_approx_params: [f64; U],
}

impl<X, D, const U: usize, const S: usize, const A: usize> Brain<X, D, U, S, A>
where
  X: State,
  D: Distribution<T = X, U = [f64; U]>,
{
  pub fn new(
    nerves: [Nerve<X>; S],
    x_distr: D,
    x_params: [f64; U],
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

  pub fn x_params(&mut self) -> &[f64; U] {
    &self.x_params
  }

  pub fn x_params_mut(&mut self) -> &mut [f64; U] {
    &mut self.x_params
  }

  fn sample_s(&mut self, x: &X) -> [f64; S] {
    let s = self
      .nerves
      .par_iter_mut()
      .map(|nerve| nerve.sample_s(x))
      .collect::<Vec<_>>();

    transmute(s)
  }

  fn bayes_est_x(&mut self) -> Result<&[f64; U], Box<dyn Error>> {
    Ok(&self.x_approx_params)
  }

  fn bayes_opt_a(&mut self) -> Result<[bool; A], Box<dyn Error>> {
    Ok([true; A])
  }

  fn free_energy(&self, s: &[f64; S]) -> Result<f64, Box<dyn Error>> {
    Ok()
  }

  fn learn_delta_free_energy(&mut self) -> Result<(), Box<dyn Error>> {
    Ok(())
  }

  pub fn act<E>(&mut self, env: &mut E) -> Result<(), Box<dyn Error>>
  where
    E: Environment<X, A>,
  {
    let x = env.state();
    let s = self.sample_s(x);

    self.bayes_est_x()?;

    let free_energy = self.free_energy(&s)?;

    let a = self.bayes_opt_a()?;
    let x = env.transition(a);
    let s = self.sample_s(x);

    let delta_free_energy = self.free_energy(&s)? - free_energy;

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
