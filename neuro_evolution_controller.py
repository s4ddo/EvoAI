# Third-party libraries
import mujoco
import numpy as np
from ariel.simulation.tasks.gait_learning import xy_displacement
import ctypes
import matplotlib.pyplot as plt


def init_genome(input_size, hidden_size, output_size):
    hidden1_size = hidden2_size = hidden_size
    n_params = (
        input_size * hidden1_size + hidden1_size + 
        hidden1_size * hidden2_size + hidden2_size + 
        hidden2_size * output_size + output_size
    )
    return np.random.uniform(-1, 1, size=n_params)


def decode_genome(genome, input_size, hidden_size, output_size):
    """Turn flat genome into weights and biases for 2 hidden layers."""

    hidden1_size = hidden2_size = hidden_size

    idx = 0

    # input → hidden1
    w1 = genome[idx: idx + input_size * hidden1_size].reshape(input_size, hidden1_size)
    idx += input_size * hidden1_size
    b1 = genome[idx: idx + hidden1_size]
    idx += hidden1_size

    # hidden1 → hidden2
    w2 = genome[idx: idx + hidden1_size * hidden2_size].reshape(hidden1_size, hidden2_size)
    idx += hidden1_size * hidden2_size
    b2 = genome[idx: idx + hidden2_size]
    idx += hidden2_size

    # hidden2 → output
    w3 = genome[idx: idx + hidden2_size * output_size].reshape(hidden2_size, output_size)
    idx += hidden2_size * output_size
    b3 = genome[idx: idx + output_size]

    return (w1, b1, w2, b2, w3, b3)


def forward_nn(x, genome, input_size, hidden_size, output_size):
    """2-hidden-layer NN: tanh activations for hidden layers, linear output."""

    w1, b1, w2, b2, w3, b3 = decode_genome(genome, input_size, hidden_size, output_size)
    h1 = np.tanh(np.dot(x, w1) + b1)
    h2 = np.tanh(np.dot(h1, w2) + b2)
    o = np.tanh(np.dot(h2, w3) + b3 ) # linear output (no activation)
    return o

def nn_control(model, data, to_track, genome, history, input_size, hidden_size, output_size):
    # Example state: joint positions + velocities
    qvel = data.qpos  # joint velocities

    action = forward_nn(qvel, genome, input_size, hidden_size, output_size)
    
    delta = 0.05
    data.ctrl += action * delta
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)
    history.append(to_track[0].xpos.copy())

from multiprocessing import Pool, cpu_count

# --- helper function for pool (must be top-level) ---
def _evaluate_star(args):
    return evaluate_one(*args)


def fitness_function(target_position, history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = target_position
    xc, yc, zc = history[-1]

    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )

    return cartesian_distance

import time as t
def evaluate_one(genome, robot_core_func, world_func, time, input_size, hidden_size, output_size, spawn_pos):
    mujoco.set_mjcb_control(None)
    world = world_func()
    world.spawn(robot_core_func().spec, spawn_pos)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    history = []
    mujoco.set_mjcb_control(
        lambda m, d: nn_control(m, d, to_track, genome, history, input_size, hidden_size, output_size)
    )

    start_time = t.time()

    while data.time < time: # and (t.time() - start_time) < 15:
        mujoco.mj_step(model, data)

    distance_to_goal = fitness_function([5,0,0.5], history)
    return distance_to_goal


def evaluate(population, robot_core_func, world_func, time, input_size, hidden_size, output_size, spawn_pos, n_workers=None):
    if n_workers is None:
        n_workers = min(len(population), cpu_count() - 1)

    results_fitness = []
    total = len(population)

    with Pool(n_workers) as pool:
        args = [
            (genome, robot_core_func, world_func, time, input_size, hidden_size, output_size, spawn_pos)
            for genome in population
        ]

        for i, result in enumerate(pool.imap_unordered(_evaluate_star, args), 1):
            results_fitness.append(result)
            print(f"Processed genome {i}/{total}. Fitness: {result}")

    return results_fitness


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
            offspring.append(np.array(new_baby))



    offspring = np.array(offspring)
    return offspring

def mutate(x, mutation_rate):
    """Apply mutation to an individual."""

    for i in x:
        for a in range(len(i)):
            if np.random.random() > mutation_rate:
                continue

            i[a] += np.random.normal(-0.1,0.1)

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

def graph_results(progress, title="Progress"):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(progress, 'm-', label='Progress Towards Target')
    plt.ylabel("Progress")
    plt.ylim(bottom = 0)
    plt.legend()
    plt.grid(True)

def main(robot_core_func, 
         world_func, 
         spawn_pos      = [-0.8, 0, 0.1], 
         time           = 30, 
         population     = 10, 
         generations    = 50, 
         p_crossover    = 0.5, 
         m_rate         = 0.1,
         hidden_size    = 16,
         graph          = False):

    mujoco.set_mjcb_control(None)
    world       = world_func()
    world.spawn(robot_core_func().spec, spawn_pos)
    model       = world.spec.compile()
    data        = mujoco.MjData(model)

    input_size  = len(data.qpos)  # e.g., 6 positions + 6 velocities  
    hidden_size = hidden_size
    output_size = model.nu

    population  = [init_genome(input_size, hidden_size, output_size) for _ in range(population)]
    population_fitness = evaluate(population, robot_core_func, world_func, time, input_size, hidden_size, output_size, spawn_pos)

    idx = np.argmin(population_fitness) 

    f_best_current = [population_fitness[idx]]

    x0_best = population[idx]
    f0_best = population_fitness[idx]
    
    x_best = [x0_best]
    f_best = [f0_best]

    for _ in range(generations):
        if _ % 5 == 0:
            print(f"Generation {_}")
        parents, parents_f             = parent_selection(population, population_fitness)
        offsprings                     = crossover(parents, p_crossover)
        
        offsprings                     = mutate(offsprings, m_rate)
        f_offspring                    = evaluate(offsprings, robot_core_func, world_func, time, input_size, hidden_size, output_size, spawn_pos)
        population, population_fitness = survivior_selection(
            parents, parents_f, offsprings, f_offspring
        )

        idx = np.argmin(population_fitness) # !!! SWITCHED FROM .argmax() to .argmin() FOR MINIMIZATION !!!
        xi_best = population[idx]
        fi_best = population_fitness[idx]

        f_best_current.append(fi_best)

        if fi_best < f_best[-1]: # !!! SWITCHED FROM > to < FOR MINIMIZATION !!!
            x_best.append(xi_best)
            f_best.append(fi_best)
        else:
            x_best.append(x_best[-1])
            f_best.append(f_best[-1])

    if graph:
        graph_results(f_best, "Best of all runs")
        graph_results(f_best_current, "Best in each run")
    
    return x_best[-1],f_best[-1]
