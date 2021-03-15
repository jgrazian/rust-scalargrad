use rand::random;

use crate::scalar::*;

/// Type of output of call() function from [Model] implementors
enum ModelOutput<'g> {
    None,
    Scalar(Scalar<'g>),
    Vector(Vec<Scalar<'g>>),
}

/// Abstraction for different types made up of [Scalars](Scalar).
trait Model {
    fn call<'a>(&'a self, x: &[Scalar<'a>]) -> ModelOutput;
    fn parameters(&self) -> Vec<&Scalar>;
    fn zero_grad(&self) {
        self.parameters().iter().for_each(|p| p.set_grad(0.0));
    }
}

impl ScalarGraph {
    /// Create a new [Neuron].
    pub fn neuron(&self, nin: usize, relu: bool) -> Neuron {
        Neuron {
            w: (0..(nin + 1))
                .into_iter()
                .map(|_| self.scalar(2.0 * random::<f64>() - 1.0))
                .collect(),
            relu,
        }
    }

    /// Create a new [Layer].
    pub fn layer(&self, nin: usize, nout: usize, relu: bool) -> Layer {
        Layer {
            neurons: (0..nout)
                .into_iter()
                .map(|_| self.neuron(nin, relu))
                .collect(),
        }
    }

    /// Create a new [MLP].
    pub fn mlp(&self, nin: usize, nouts: &[usize], relu: bool) -> MLP {
        let sz: &[usize] = &[&[nin], nouts].concat();
        MLP {
            layers: (0..nouts.len())
                .into_iter()
                .map(|i| self.layer(sz[i], sz[i + 1], (i != nouts.len() - 1) && relu))
                .collect(),
        }
    }
}

/// A single perceptron
///
/// Holds nin + 1 values, for nin weights and 1 bias value.
/// Not typically instantiated by itself. Usually issued via [ScalarGraph].
///
/// # Examples
///
/// ```
/// use scalargrad::ScalarGraph;
///
/// let mut g = ScalarGraph::new();
///
/// let n = g.neuron(4, true); // <- Is a Neuron
/// assert_eq!(n.w().len(), 4); // 4 weights
/// assert_eq!(n.b().len(), 1); // 1 bias
/// ```
pub struct Neuron<'g> {
    w: Vec<Scalar<'g>>,
    relu: bool,
}

impl Neuron<'_> {
    fn w(&self) -> &[Scalar] {
        &self.w[1..]
    }

    fn b(&self) -> Scalar {
        self.w[0]
    }
}

impl Model for Neuron<'_> {
    fn call<'a>(&'a self, x: &[Scalar<'a>]) -> ModelOutput {
        let act: Scalar = self
            .w()
            .iter()
            .zip(x)
            .map(|(w, x)| w * x)
            .reduce(|a, b| a + b)
            .unwrap()
            + self.b();

        if self.relu {
            ModelOutput::Scalar(act.relu())
        } else {
            ModelOutput::Scalar(act)
        }
    }

    fn parameters(&self) -> Vec<&Scalar> {
        self.w.iter().map(|wi| wi).collect()
    }
}

/// A collection of Neurons
///
/// Holds nout neurons, each of which have (nin + 1) weights.
/// Not typically instantiated by itself. Usually issued via [ScalarGraph].
///
/// # Examples
///
/// ```
/// use scalargrad::ScalarGraph;
///
/// let mut g = ScalarGraph::new();
///
/// let l = g.layer(4, 2, true); // <- Is a Layer of 2 Neurons
/// assert_eq!(l.parameters().len(), 10); // nin * (nout + 1) parameters
/// ```
pub struct Layer<'g> {
    neurons: Vec<Neuron<'g>>,
}

impl Model for Layer<'_> {
    fn call<'a>(&'a self, x: &[Scalar<'a>]) -> ModelOutput {
        ModelOutput::Vector(
            self.neurons
                .iter()
                .map(|n| match n.call(x) {
                    ModelOutput::Scalar(s) => s,
                    _ => unreachable!("Expected Scalar variant"),
                })
                .collect(),
        )
    }

    fn parameters(&self) -> Vec<&Scalar> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

/// A multi-layer perceptron or series of fully connected layers.
///
/// Holds nout.len() + 1 layers.
/// Not typically instantiated by itself. Usually issued via [ScalarGraph].
///
/// # Examples
/// The below example creates a MLP with 4 inputs, 2 hidden layers of 3 neurons and 1 output value.
///
/// ```
/// use scalargrad::ScalarGraph;
///
/// let mut g = ScalarGraph::new();
///
/// let m = g.mlp(4, &[3, 3, 1], true); // <- Is a MLP
/// ```
pub struct MLP<'g> {
    layers: Vec<Layer<'g>>,
}

impl Model for MLP<'_> {
    fn call<'a>(&'a self, x: &[Scalar<'a>]) -> ModelOutput {
        let mut out = ModelOutput::None;

        for layer in &self.layers {
            out = match out {
                ModelOutput::Scalar(s) => layer.call(&[s]),
                ModelOutput::Vector(v) => layer.call(&v),
                _ => layer.call(x),
            };
        }
        out
    }

    fn parameters(&self) -> Vec<&Scalar> {
        self.layers.iter().flat_map(|n| n.parameters()).collect()
    }
}
