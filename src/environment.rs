use std::fmt::Debug;

pub trait Environment<X, const A: usize>
where
  X: State,
{
  fn state(&self) -> &X;
  fn transition(&mut self, a: [bool; A]) -> &X;
}

pub trait State: Clone + Debug + Send + Sync {}

impl<T> State for T where T: Clone + Debug + Send + Sync {}

#[cfg(test)]
mod tests {
  #[test]
  fn it_works() {
    assert_eq!(2 + 2, 4);
  }
}
