
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

    return construct_mjspec_from_graph(robot_graph)


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

def parent_selection(x, f):
    x_parents, f_parents = [],[]
    
    total_f              = np.sum(f)
    
    s                    = 0.000000000001
    max_probs            = np.array([i / (total_f + s) for i in f])    
    min_probs            = 1 / (max_probs + s)
    min_probs_normalized = min_probs / (min_probs.sum() + s)
    
    for _ in range(int(len(x)/2)):
        parent_indices = np.random.choice(len(x), size=2, replace=True, p=min_probs_normalized)
        x_parents.append([x[i] for i in parent_indices])
        f_parents.append([f[i] for i in parent_indices])

    return x_parents, f_parents

def crossover(x_parents, p_crossover):
    offspring = []
    for pair in x_parents:
        if np.random.random() > p_crossover:
            offspring.extend(pair)
            continue
        
        p1, p2 = pair
        
        # 1) Get crossover points
        crossover_point_1 = np.random.randint(0,int(len(p1)))
        crossover_point_2 = np.random.randint(crossover_point_1,len(p1))
        
        # 2) Generate two offsprings, px represents parent of which we get the strand from. the remaining gene we get from p2
        for px, py in [(p1,p2),(p2,p1)]:
            strand_1 = px[crossover_point_1:crossover_point_2]
            
            new_baby = list(py[:crossover_point_1]) + list(strand_1) + list(py[crossover_point_2:])
            print(len(p1), len(new_baby))
            offspring.append(np.array(new_baby))



    offspring = np.array(offspring)
    return offspring

def mutate(x, mutation_rate):
    """Apply mutation to an individual."""

    for i in x:
        for a in range(len(i)):
            if np.random.random() > mutation_rate:
                continue

            i[a] = np.random.uniform(low=-np.pi/2, high=np.pi/2)

    return x

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

def evaluate_population(population):
    res = []
    for robot_genotype in population:
        robot_core_func = lambda : make_robot(robot_genotype)

        robot_fitness = evolve_robot_controller(robot_core_func = robot_core_func, 
                                                world_func      = OlympicArena, 
                                                spawn_pos       = [-0.8, -0, 0.1],
                                                time            = 30,
                                                population      = 10,
                                                p_crossover     = 0.5,
                                                m_rate          = 0.5)
        
        res.append(robot_fitness)
    return res

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

def main(time = 30, n_population = 10, generations = 50, p_crossover = 0.5, m_rate = 0.1):
    population         = init_population(n_population)
    population_fitness = evaluate_population(population)

    idx = np.argmin(population_fitness) 
    x0_best, f0_best = population[idx],  population_fitness[idx]
    x_best,  f_best  = [x0_best], [f0_best]

    for _ in range(generations):
        parents, parents_f             = parent_selection(population, population_fitness)
        offsprings                     = crossover(parents, p_crossover)
        offsprings                     = mutate(offsprings, m_rate)
        f_offspring                    = evaluate_population(offsprings)
        population, population_fitness = survivior_selection(
            parents, parents_f, offsprings, f_offspring
        )

        store_best(population, population_fitness, x_best, f_best)
    
    return x_best, f_best
