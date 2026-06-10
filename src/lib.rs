use pyo3::prelude::*;

mod genetics;
mod network;

use genetics::Population as RustPopulation;

/// Python-facing wrapper around the genetic population.
#[pyclass]
struct Population {
    inner: RustPopulation,
}

#[pymethods]
impl Population {
    /// Create a new randomly initialised population.
    ///
    /// topology : e.g. [9, 8, 3]  (input → hidden → output)
    #[new]
    #[pyo3(signature = (size, topology, mutation_rate=0.1, mutation_sigma=0.2, elite_count=2, parent_pool_frac=0.5))]
    fn new(
        size: usize,
        topology: Vec<usize>,
        mutation_rate: f32,
        mutation_sigma: f32,
        elite_count: usize,
        parent_pool_frac: f32,
    ) -> Self {
        Self {
            inner: RustPopulation::new(
                size,
                topology,
                mutation_rate,
                mutation_sigma,
                elite_count,
                parent_pool_frac,
            ),
        }
    }

    /// Forward-pass for agent `idx`. Returns [steer, gas, brake] in [-1, 1].
    fn get_actions(&self, idx: usize, inputs: Vec<f32>) -> Vec<f32> {
        self.inner.get_actions(idx, &inputs)
    }

    /// Advance one generation given a fitness score per agent.
    /// Expects len(fitnesses) == self.size.
    fn evolve(&mut self, fitnesses: Vec<f32>) {
        self.inner.evolve(&fitnesses);
    }

    #[getter]
    fn generation(&self) -> usize {
        self.inner.generation
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.individuals.len()
    }

    fn best_fitness(&self) -> f32 {
        self.inner.best_fitness()
    }

    #[getter]
    fn all_time_best_fitness(&self) -> f32 {
        self.inner.all_time_best_fitness
    }

    fn get_all_time_best_weights(&self) -> Vec<f32> {
        self.inner.all_time_best_weights.clone()
    }

    /// Restore all-time best from a checkpoint (called during resume).
    fn set_all_time_best(&mut self, weights: Vec<f32>, fitness: f32) {
        self.inner.all_time_best_weights = weights;
        self.inner.all_time_best_fitness = fitness;
    }

    /// All weights, one Vec<f32> per individual (for serialization).
    fn get_all_weights(&self) -> Vec<Vec<f32>> {
        self.inner
            .individuals
            .iter()
            .map(|ind| ind.weights.clone())
            .collect()
    }

    /// Fitnesses in the same order as get_all_weights (before next evolve call).
    fn get_fitnesses(&self) -> Vec<f32> {
        self.inner
            .individuals
            .iter()
            .map(|ind| ind.fitness)
            .collect()
    }

    /// Overwrite the weights of individual `idx` (for loading a checkpoint).
    fn set_weights_at(&mut self, idx: usize, weights: Vec<f32>) {
        assert!(idx < self.inner.individuals.len(), "idx out of bounds");
        self.inner.individuals[idx].weights = weights;
    }

    #[getter]
    fn topology(&self) -> Vec<usize> {
        self.inner.topology.clone()
    }
}

#[pymodule]
fn tmnf_evo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Population>()?;
    Ok(())
}
