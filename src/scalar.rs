use std::collections::HashSet;
use std::ops;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::graph::*;

struct ScalarGraph {
    inner: RwLock<Graph<ScalarData>>,
}

impl ScalarGraph {
    pub fn with<FN, R>(f: FN) -> R
    where
        FN: FnOnce(&mut ScalarGraph) -> R,
    {
        let mut g = ScalarGraph {
            inner: RwLock::new(Graph::new()),
        };
        f(&mut g)
    }

    fn read(&self) -> RwLockReadGuard<Graph<ScalarData>> {
        self.inner.read().unwrap()
    }

    fn write(&self) -> RwLockWriteGuard<Graph<ScalarData>> {
        self.inner.write().unwrap()
    }

    pub fn scalar(&self, data: f64) -> Scalar {
        let id = self.inner.write().unwrap().push(ScalarData {
            data,
            grad: 0.0,
            op: Op::None,
        });

        Scalar { id, graph: &self }
    }

    fn scalar_op(&self, data: f64, op: Op) -> Scalar {
        let id = self.inner.write().unwrap().push(ScalarData {
            data,
            grad: 0.0,
            op,
        });

        Scalar { id, graph: &self }
    }
}

#[derive(Copy, Clone, Debug)]
enum Op {
    None,
    Add(usize, usize),
    Mul(usize, usize),
    Pow(usize, f64),
    ReLU(usize),
}

impl Op {
    fn backward(graph: &ScalarGraph, id: usize) {
        let mut con = graph.write();

        let grad = con.node(id).grad;

        match con.node(id).op {
            Op::Add(s, o) => {
                con.node_mut(s).grad += grad;
                con.node_mut(o).grad += grad;
            }
            Op::Mul(s, o) => {
                con.node_mut(s).grad += con.node(o).data * grad;
                con.node_mut(o).grad += con.node(s).data * grad;
            }
            Op::Pow(s, o) => {
                con.node_mut(s).grad += (o * con.node(s).data.powf(o - 1.0)) * grad;
            }
            Op::ReLU(s) => {
                con.node_mut(s).grad += if con.node(id).data > 0.0 { grad } else { 0.0 };
            }
            Op::None => {}
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ScalarData {
    data: f64,
    grad: f64,
    op: Op,
}

#[derive(Copy, Clone)]
struct Scalar<'g> {
    id: usize,
    graph: &'g ScalarGraph,
}

impl Scalar<'_> {
    fn with_op(&self, data: f64, op: Op) -> Self {
        self.graph.scalar_op(data, op)
    }

    pub fn data(&self) -> f64 {
        self.graph.read().node(self.id).data
    }

    fn set_data(&self, data: f64) {
        self.graph.write().node_mut(self.id).data = data
    }

    pub fn grad(&self) -> f64 {
        self.graph.read().node(self.id).grad
    }

    fn set_grad(&self, grad: f64) {
        self.graph.write().node_mut(self.id).grad = grad
    }

    pub fn pow(&self, other: f64) -> Self {
        self.with_op(self.data().powf(other), Op::Pow(self.id, other))
    }

    pub fn relu(&self) -> Self {
        self.with_op(
            if self.data() > 0.0 { self.data() } else { 0.0 },
            Op::ReLU(self.id),
        )
    }

    fn topo(&self) -> Vec<usize> {
        let mut sort = Vec::new();
        let mut visited = HashSet::new();

        fn dfs(
            graph: &ScalarGraph,
            id: usize,
            sort: &mut Vec<usize>,
            visited: &mut HashSet<usize>,
        ) {
            if !visited.contains(&id) {
                visited.insert(id);
                match graph.read().node(id).op {
                    Op::Add(s, o) => {
                        dfs(graph, s, sort, visited);
                        dfs(graph, o, sort, visited);
                    }
                    Op::Mul(s, o) => {
                        dfs(graph, s, sort, visited);
                        dfs(graph, o, sort, visited);
                    }
                    Op::Pow(s, _) => dfs(graph, s, sort, visited),
                    Op::ReLU(s) => dfs(graph, s, sort, visited),
                    Op::None => {}
                }
                sort.push(id);
            }
        }

        dfs(&self.graph, self.id, &mut sort, &mut visited);
        sort
    }

    pub fn backward(&self) {
        let sorted = self.topo();

        self.set_grad(1.0);
        for id in sorted.iter().rev() {
            Op::backward(&self.graph, *id);
        }
    }
}

// ---------------------------------
// -------------- ADD --------------
// ---------------------------------
impl ops::Add for Scalar<'_> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.graph
            .scalar_op(self.data() + other.data(), Op::Add(self.id, other.id))
    }
}
impl<'a> ops::Add for &Scalar<'a> {
    type Output = Scalar<'a>;
    fn add(self, other: Self) -> Scalar<'a> {
        self.graph
            .scalar_op(self.data() + other.data(), Op::Add(self.id, other.id))
    }
}
impl ops::Add<f64> for Scalar<'_> {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        let other = self.graph.scalar(rhs);
        self.graph
            .scalar_op(self.data() + other.data(), Op::Add(self.id, other.id))
    }
}
impl<'a> ops::Add<Scalar<'a>> for f64 {
    type Output = Scalar<'a>;
    fn add(self, rhs: Scalar<'a>) -> Scalar<'a> {
        let lhs = rhs.graph.scalar(self);
        lhs.graph
            .scalar_op(lhs.data() + rhs.data(), Op::Add(lhs.id, rhs.id))
    }
}

// ---------------------------------
// -------------- MUL --------------
// ---------------------------------
impl ops::Mul for Scalar<'_> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.graph
            .scalar_op(self.data() * other.data(), Op::Mul(self.id, other.id))
    }
}
impl<'a> ops::Mul for &Scalar<'a> {
    type Output = Scalar<'a>;
    fn mul(self, other: Self) -> Scalar<'a> {
        self.graph
            .scalar_op(self.data() * other.data(), Op::Mul(self.id, other.id))
    }
}
impl ops::Mul<f64> for Scalar<'_> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        let other = self.graph.scalar(rhs);
        self.graph
            .scalar_op(self.data() * other.data(), Op::Mul(self.id, other.id))
    }
}
impl<'a> ops::Mul<Scalar<'a>> for f64 {
    type Output = Scalar<'a>;
    fn mul(self, rhs: Scalar<'a>) -> Scalar<'a> {
        let lhs = rhs.graph.scalar(self);
        lhs.graph
            .scalar_op(lhs.data() * rhs.data(), Op::Mul(lhs.id, rhs.id))
    }
}
impl<'a> ops::Mul<&Scalar<'a>> for f64 {
    type Output = Scalar<'a>;
    fn mul(self, rhs: &Scalar<'a>) -> Scalar<'a> {
        let lhs = rhs.graph.scalar(self);
        lhs.graph
            .scalar_op(lhs.data() * rhs.data(), Op::Mul(lhs.id, rhs.id))
    }
}

// ---------------------------------
// -------------- SUB --------------
// ---------------------------------
impl ops::Sub for Scalar<'_> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}
impl<'a> ops::Sub for &Scalar<'a> {
    type Output = Scalar<'a>;
    fn sub(self, other: Self) -> Scalar<'a> {
        self + (&-other)
    }
}
impl ops::Sub<f64> for Scalar<'_> {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        self + (-rhs)
    }
}
impl<'a> ops::Sub<Scalar<'a>> for f64 {
    type Output = Scalar<'a>;
    fn sub(self, rhs: Scalar<'a>) -> Scalar<'a> {
        self + (-rhs)
    }
}

// ---------------------------------
// -------------- DIV --------------
// ---------------------------------
impl ops::Div for Scalar<'_> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self * other.pow(-1.0)
    }
}
impl<'a> ops::Div for &Scalar<'a> {
    type Output = Scalar<'a>;
    fn div(self, other: Self) -> Scalar<'a> {
        self * &other.pow(-1.0)
    }
}
impl ops::Div<f64> for Scalar<'_> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        self * rhs.powf(-1.0)
    }
}
impl<'a> ops::Div<Scalar<'a>> for f64 {
    type Output = Scalar<'a>;
    fn div(self, rhs: Scalar<'a>) -> Scalar<'a> {
        self * rhs.pow(-1.0)
    }
}

// ---------------------------------
// -------------- NEG --------------
// ---------------------------------
impl ops::Neg for Scalar<'_> {
    type Output = Self;
    fn neg(self) -> Self {
        -1.0 * self
    }
}
impl<'a> ops::Neg for &Scalar<'a> {
    type Output = Scalar<'a>;
    fn neg(self) -> Scalar<'a> {
        -1.0 * self
    }
}

#[cfg(test)]
mod tests {
    use super::ScalarGraph as sg;

    #[test]
    fn add() {
        sg::with(|g| {
            let a = g.scalar(3.0);
            let b = g.scalar(4.0);
            let c = a + b;
            assert_eq!(c.data(), 7.0);
            c.backward();
            assert_eq!(a.grad(), 1.0);
        });
    }

    #[test]
    fn mul() {
        sg::with(|g| {
            let a = g.scalar(3.0);
            let b = g.scalar(4.0);
            let c = a * b;
            assert_eq!(c.data(), 12.0);
            c.backward();
            assert_eq!(a.grad(), 4.0);
        });
    }

    #[test]
    fn pow() {
        sg::with(|g| {
            let a = g.scalar(3.0);
            let b = a.pow(2.0);
            assert_eq!(b.data(), 9.0);
            b.backward();
            assert_eq!(a.grad(), 6.0);
        });
    }

    #[test]
    fn relu() {
        sg::with(|g| {
            assert_eq!(g.scalar(3.0).relu().data(), 3.0);
            assert_eq!(g.scalar(-2.0).relu().data(), 0.0);
        });
    }

    #[test]
    fn backwards() {
        sg::with(|g| {
            let a = g.scalar(1.5); // 1.5
            let b = g.scalar(-4.0); // -4.0
            let c = a.pow(3.0) / 5.0; // a^3 / 5
            let d = &b.pow(2.0).relu() + &c; // Relu(b^2) + c

            d.backward();
            assert_eq!(d.data(), 16.675);
            assert_eq!(a.grad(), 1.35);
            assert_eq!(b.grad(), -8.0);
        });
    }
}
