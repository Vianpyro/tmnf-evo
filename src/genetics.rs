use rand::prelude::*;
use rand_distr::{Distribution, Normal};

use crate::network::Network;

pub struct Individual {
    pub weights: Vec<f32>,
    pub fitness: f32,
}

pub struct Population {
    pub topology: Vec<usize>,
    pub individuals: Vec<Individual>,
    pub generation: usize,
    mutation_rate: f32,
    mutation_sigma: f32,
    elite_count: usize,
    tournament_k: usize,
    parent_pool_size: usize,
    // Best individual ever seen across all generations — never overwritten.
    // Slot 0 of every next generation is always this individual.
    pub all_time_best_fitness: f32,
    pub all_time_best_weights: Vec<f32>,
}

impl Population {
    pub fn new(
        size: usize,
        topology: Vec<usize>,
        mutation_rate: f32,
        mutation_sigma: f32,
        elite_count: usize,
        parent_pool_frac: f32,
    ) -> Self {
        assert!(
            size > elite_count,
            "elite_count must be less than population size"
        );

        // At least elite_count + 1 parents so crossover always has choices.
        let parent_pool_size = ((size as f32 * parent_pool_frac) as usize)
            .max(elite_count + 1)
            .min(size);

        let weight_count = Network::weight_count(&topology);
        let mut rng = rand::rng();

        // Xavier-ish init: uniform in [-1, 1].
        let individuals = (0..size)
            .map(|_| Individual {
                weights: (0..weight_count)
                    .map(|_| rng.random_range(-1.0f32..1.0f32))
                    .collect(),
                fitness: 0.0,
            })
            .collect();

        Self {
            topology,
            individuals,
            generation: 0,
            mutation_rate,
            mutation_sigma,
            elite_count,
            tournament_k: 3,
            parent_pool_size,
            all_time_best_fitness: f32::NEG_INFINITY,
            all_time_best_weights: Vec::new(),
        }
    }

    /// Forward-pass for a single agent. Returns output activations.
    pub fn get_actions(&self, idx: usize, inputs: &[f32]) -> Vec<f32> {
        let net = Network::new(self.topology.clone(), self.individuals[idx].weights.clone());
        net.forward(inputs)
    }

    /// Advance one generation.
    ///
    /// Assumes fitnesses.len() == self.individuals.len() and values are finite.
    pub fn evolve(&mut self, fitnesses: &[f32]) {
        assert_eq!(fitnesses.len(), self.individuals.len());

        for (ind, &f) in self.individuals.iter_mut().zip(fitnesses) {
            ind.fitness = f;
        }

        // Sort descending so elite indices are 0..elite_count.
        self.individuals.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update all-time best before building the next generation.
        if self.individuals[0].fitness > self.all_time_best_fitness {
            self.all_time_best_fitness = self.individuals[0].fitness;
            self.all_time_best_weights = self.individuals[0].weights.clone();
        }

        let pop_size = self.individuals.len();
        let normal = Normal::new(0.0f32, self.mutation_sigma).unwrap();
        let mut rng = rand::rng();
        let mut next_gen: Vec<Individual> = Vec::with_capacity(pop_size);

        // Slot 0: all-time best — guaranteed to survive every generation.
        let has_all_time = !self.all_time_best_weights.is_empty();
        if has_all_time {
            next_gen.push(Individual {
                weights: self.all_time_best_weights.clone(),
                fitness: 0.0,
            });
        }

        // Remaining elite slots: current generation's top individuals.
        let remaining_elite = if has_all_time {
            self.elite_count.saturating_sub(1)
        } else {
            self.elite_count
        };
        for i in 0..remaining_elite.min(pop_size) {
            next_gen.push(Individual {
                weights: self.individuals[i].weights.clone(),
                fitness: 0.0,
            });
        }

        // Fill remaining slots with crossover + mutation.
        while next_gen.len() < pop_size {
            let p1 = self.tournament_select(&mut rng);
            let p2 = self.tournament_select(&mut rng);
            let mut child = self.uniform_crossover(p1, p2, &mut rng);
            self.gaussian_mutate(&mut child, &normal, &mut rng);
            next_gen.push(Individual {
                weights: child,
                fitness: 0.0,
            });
        }

        self.individuals = next_gen;
        self.generation += 1;
    }

    pub fn best_fitness(&self) -> f32 {
        self.individuals
            .iter()
            .map(|i| i.fitness)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    // ------------------------------------------------------------------
    // Private
    // ------------------------------------------------------------------

    /// Tournament selection: returns the index of the winner.
    fn tournament_select(&self, rng: &mut ThreadRng) -> usize {
        (0..self.tournament_k)
            .map(|_| rng.random_range(0..self.parent_pool_size))
            .max_by(|&a, &b| {
                self.individuals[a]
                    .fitness
                    .partial_cmp(&self.individuals[b].fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    }

    /// Uniform crossover: each gene is taken from p1 or p2 with equal probability.
    fn uniform_crossover(&self, p1: usize, p2: usize, rng: &mut ThreadRng) -> Vec<f32> {
        self.individuals[p1]
            .weights
            .iter()
            .zip(&self.individuals[p2].weights)
            .map(|(&w1, &w2)| if rng.random::<bool>() { w1 } else { w2 })
            .collect()
    }

    /// Per-weight Gaussian mutation with probability mutation_rate.
    fn gaussian_mutate(&self, weights: &mut Vec<f32>, normal: &Normal<f32>, rng: &mut ThreadRng) {
        for w in weights.iter_mut() {
            if rng.random::<f32>() < self.mutation_rate {
                *w += normal.sample(rng);
            }
        }
    }
}
