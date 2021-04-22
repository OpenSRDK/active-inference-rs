use opensrdk_probability::RandomVariable;

pub trait Environment<X, const A: usize>
where
  X: State,
{
  fn state(&self) -> &X;
  fn transition(&mut self, a: [bool; A]) -> &X;
}

pub trait State: RandomVariable {}

impl<T> State for T where T: RandomVariable {}

#[cfg(test)]
mod tests {
  #[test]
  fn it_works() {
    assert_eq!(2 + 2, 4);
  }
}
