
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from typing import TYPE_CHECKING, Any, Literal
if TYPE_CHECKING:
    from networkx import DiGraph
from pathlib import Path
import numpy as np
from ariel.simulation.environments import OlympicArena
import mujoco
from neuro_evolution_controller import main as evolve_robot_controller

RNG             = np.random.default_rng(1)
NUM_OF_MODULES  = 30
SCRIPT_NAME     = "test"
CWD             = Path.cwd()
DATA            = CWD / "__data__" / SCRIPT_NAME

DATA.mkdir(exist_ok=True)


def make_robot(genotype):
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genotype)

    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # save_graph_as_json(
    #     robot_graph,
    #     DATA / "robot_graph.json",
    # )
    return robot_graph


def init_population(population: int):
    res = []
    genotype_size = 64

    for _ in range(population):
        res.append([
            RNG.random(genotype_size).astype(np.float32),
            RNG.random(genotype_size).astype(np.float32),
            RNG.random(genotype_size).astype(np.float32)
        ])

    return res

from concurrent.futures import ProcessPoolExecutor, as_completed


def evaluate_single_robot(index, robot_genotype):
    print(f"Processing robot {index + 1} ...")
    
    graph = make_robot(robot_genotype)
    robot_core_func = lambda: construct_mjspec_from_graph(graph)
    
    robot_fitness = evolve_robot_controller(
        robot_core_func=robot_core_func,
        world_func=OlympicArena,
        spawn_pos=[-0.8, -0, 0.1],
        time=30,
        population=10,
        p_crossover=0.5,
        m_rate=0.5,
        generations=1
    )
    
    print(f"Finished robot {index + 1}")
    return index, robot_fitness


def evaluate_population(population, max_workers=4):
    results = [None] * len(population)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_single_robot, i, robot_genotype): i
            for i, robot_genotype in enumerate(population)
        }
        
        for future in as_completed(futures):
            i = futures[future]
            try:
                index, fitness = future.result()
                results[index] = fitness
            except Exception as e:
                print(f"Error evaluating robot {i + 1}: {e}")
                results[i] = None
                
    print("✅ All robots processed.")
    return results


def parent_selection(population, f):
    """Selects parents from the population based on their fitness."""
    x_parents, f_parents = [], []
    
    # Convert fitness list to a numpy array for easier calculations
    fitness_values = np.array(f)
    
    # Since this is a minimization problem (lower fitness is better), we transform
    # the fitness scores so that lower scores get a higher probability.
    # We subtract each score from the max score to invert the ranking.
    max_fitness = np.max(fitness_values)
    
    # Add a small epsilon to ensure even the worst-performing individual (transformed score of 0)
    # has a tiny, non-zero chance of being selected.
    s = 1e-9 
    
    # Transformed fitness: The best (lowest) original scores are now the highest values.
    transformed_fitness = (max_fitness - fitness_values) + s
    
    # Normalize the positive, transformed fitness scores to get valid probabilities.
    probabilities = transformed_fitness / np.sum(transformed_fitness)

    # Loop to select pairs of parents
    for _ in range(int(len(population) / 2)):
        # Use the new, valid probabilities for selection
        parent_indices = np.random.choice(
            len(population), 
            size=2, 
            replace=True, 
            p=probabilities
        )
        x_parents.append([population[i] for i in parent_indices])
        f_parents.append([f[i] for i in parent_indices])

    return x_parents, f_parents


def crossover_basic(x_parents, p_crossover):
    offspring = []
    for pair in x_parents:
        if np.random.random() > p_crossover:
            offspring.extend(pair)
            continue
        
        p1, p2 = pair
        p1, p2 = p1.copy(), p2.copy()
        
        # 1) Get crossover points
        crossover_point = np.random.randint(0,int(len(p1)))
        
        _temp = p1[crossover_point]
        p1[crossover_point] = p2[crossover_point]
        p2[crossover_point] = _temp

        offspring.append(p1)
        offspring.append(p2)

    offspring = np.array(offspring)
    return offspring

def crossover(x_parents, p_crossover, crossover_mode = "BASIC"):
    if crossover_mode == "BASIC":
        return crossover_basic(x_parents, p_crossover)

    return crossover_basic(x_parents, p_crossover)

def mutate(population, mutation_rate):
    """Apply mutation to an individual."""

    for gene in population:
        for strand in gene:
            for i in range(len(strand)):
                if np.random.random() > mutation_rate:
                    continue
                
                strand[i] = np.random.random() # TODO: Switch with another index instead

    return population

def survivior_selection(x, f, x_offspring, f_offspring):
    """Select the survivors, for the population of the next generation. Returns a list of survivors and their fitness values."""

    # 1) This one looks slightly weird since in the provided code the x and f is passed as a parent pair so it needs to be flattend
    _x = []
    for i in x: _x.extend(i)
    
    _f = []
    for i in f: _f.extend(i)
    
    x_offspring = x_offspring.tolist() # For simplicity

    # 2) Get combine population
    combined_x = np.array(_x + x_offspring)
    combined_f = np.array(_f + f_offspring)
    
    # 3) Sort based on best performing
    sorted_indices = np.argsort(combined_f)
    
    # 4) Get n best performing
    n = len(_x)
    x = combined_x[sorted_indices[:n]]
    f = combined_f[sorted_indices[:n]]

    return x, f

def store_best(population, population_fitness, x_best, f_best):
    idx = np.argmin(population_fitness) # !!! SWITCHED FROM .argmax() to .argmin() FOR MINIMIZATION !!!
    xi_best = population[idx]
    fi_best = population_fitness[idx]
    print(f"Best one is {fi_best}")
    if fi_best < f_best[-1]: # !!! SWITCHED FROM > to < FOR MINIMIZATION !!!
        x_best.append(xi_best)
        f_best.append(fi_best)
    else:
        x_best.append(x_best[-1])
        f_best.append(f_best[-1])

def main(n_population = 10, generations = 50, p_crossover = 0.5, m_rate = 0.1):
    population         = init_population(n_population)
    population_fitness = evaluate_population(population)

    idx              = np.argmin(population_fitness) 
    x0_best, f0_best = population[idx],  population_fitness[idx]
    x_best,  f_best  = [x0_best], [f0_best]

    for _ in range(generations):
        print(f"============= Generation {_ + 1} =============")

        parents, parents_f             = parent_selection(population, population_fitness)
        offsprings                     = crossover(parents, p_crossover)
        offsprings                     = mutate(offsprings, m_rate)
        f_offspring                    = evaluate_population(offsprings)
        population, population_fitness = survivior_selection(
            parents, parents_f, offsprings, f_offspring
        )

        store_best(population, population_fitness, x_best, f_best)
    
    return x_best, f_best
