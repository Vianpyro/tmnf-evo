/// Dense feedforward neural network with tanh activations.
///
/// Weight layout (flat Vec<f32>):
///   For each layer l (0..topology.len()-1):
///     n_in = topology[l], n_out = topology[l+1]
///     Per neuron j: [w[j][0], ..., w[j][n_in-1], bias[j]]  (n_in+1 values)
///     Total per layer: n_out * (n_in + 1)
pub struct Network {
    topology: Vec<usize>,
    weights: Vec<f32>,
}

impl Network {
    /// Total number of parameters for a given topology.
    pub fn weight_count(topology: &[usize]) -> usize {
        topology.windows(2).map(|w| w[1] * (w[0] + 1)).sum()
    }

    pub fn new(topology: Vec<usize>, weights: Vec<f32>) -> Self {
        debug_assert_eq!(
            weights.len(),
            Self::weight_count(&topology),
            "weight count mismatch"
        );
        Self { topology, weights }
    }

    /// Forward pass. Returns output activations (tanh everywhere).
    pub fn forward(&self, inputs: &[f32]) -> Vec<f32> {
        debug_assert_eq!(inputs.len(), self.topology[0]);

        let mut current = inputs.to_vec();
        let mut offset = 0;

        for layer in self.topology.windows(2) {
            let (n_in, n_out) = (layer[0], layer[1]);
            let stride = n_in + 1; // weights + bias per neuron
            let mut next = vec![0.0f32; n_out];

            for j in 0..n_out {
                let base = offset + j * stride;
                // Bias is stored after the n_in weights.
                let mut sum = self.weights[base + n_in];
                for i in 0..n_in {
                    sum += self.weights[base + i] * current[i];
                }
                next[j] = sum.tanh();
            }

            offset += n_out * stride;
            current = next;
        }

        current
    }
}
