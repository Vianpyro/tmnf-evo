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
    #[pyo3(signature = (size, topology, mutation_rate=0.1, mutation_sigma=0.2, elite_count=2))]
    fn new(
        size: usize,
        topology: Vec<usize>,
        mutation_rate: f32,
        mutation_sigma: f32,
        elite_count: usize,
    ) -> Self {
        Self {
            inner: RustPopulation::new(size, topology, mutation_rate, mutation_sigma, elite_count),
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
}

#[pymodule]
fn tmnf_evo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Population>()?;
    Ok(())
}
