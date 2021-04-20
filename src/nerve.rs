use crate::State;
use opensrdk_probability::rand::prelude::*;
use opensrdk_probability::Distribution;
use opensrdk_probability::*;
use std::error::Error;

pub struct Nerve<X>
where
  X: State,
{
  params: NormalParams,
  interface: Box<dyn Fn(&X) -> f64>,
  rng: StdRng,
}

impl<X> Nerve<X>
where
  X: State,
{
  pub fn new(interface: Box<dyn Fn(&X) -> f64>, rng: StdRng) -> Self {
    Self {
      params: NormalParams::new(0.0, 1.0).unwrap(),
      interface,
      rng,
    }
  }

  pub fn set_presicion(&mut self, sigma: f64) -> Result<&mut Self, Box<dyn Error>> {
    self.params = NormalParams::new(0.0, sigma)?;

    Ok(self)
  }

  pub fn sample_s(&mut self, x: &X) -> f64 {
    (self.interface)(x) + Normal.sample(&self.params, &mut self.rng).unwrap_or(0.0)
  }
}

unsafe impl<X> Send for Nerve<X> where X: State {}
unsafe impl<X> Sync for Nerve<X> where X: State {}
