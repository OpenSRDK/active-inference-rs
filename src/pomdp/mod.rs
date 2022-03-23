use opensrdk_probability::RandomVariable;

pub mod world_model;

pub use world_model::*;

pub trait POMDP<X, A>
where
    X: RandomVariable,
    A: RandomVariable,
{
    fn state(&self) -> &X;
    fn transition(&mut self, a: &A) -> &X;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
