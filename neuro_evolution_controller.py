# Third-party libraries
import mujoco
import numpy as np
from ariel.simulation.tasks.gate_learning import xy_displacement
import ctypes

def init_genome(input_size, hidden_size, output_size):
    # Flattened [input→hidden, hidden bias, hidden→output, output bias]
    n_params = input_size*hidden_size + hidden_size + hidden_size*output_size + output_size
    return np.random.uniform(-1, 1, size=n_params)

def decode_genome(genome, input_size, hidden_size, output_size):
    """Turn flat genome into weights and biases."""
    idx = 0
    
    # input→hidden
    w1 = genome[idx: idx + input_size*hidden_size].reshape(input_size, hidden_size)
    idx += input_size*hidden_size
    b1 = genome[idx: idx + hidden_size]
    idx += hidden_size

    # hidden→output
    w2 = genome[idx: idx + hidden_size*output_size].reshape(hidden_size, output_size)
    idx += hidden_size*output_size
    b2 = genome[idx: idx + output_size]

    return (w1, b1, w2, b2)

def forward_nn(x, genome, input_size, hidden_size, output_size):
    """Simple 1-hidden-layer NN with tanh activation."""
    w1, b1, w2, b2 = decode_genome(genome, input_size, hidden_size, output_size)
    h = np.tanh(np.dot(x, w1) + b1)
    o = np.tanh(np.dot(h, w2) + b2)
    return o

def nn_control(model, data, to_track, genome, history, input_size, hidden_size, output_size):
    # Example state: joint positions + velocities
    qvel = data.qvel  # joint velocities

    action = forward_nn(qvel, genome, input_size, hidden_size, output_size)
    
    delta = 0.05
    data.ctrl += action * delta
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)
    history.append(to_track[0].xpos.copy())

def evaluate(population, robot_core_func, world_func, time, input_size, hidden_size, output_size):
    results_fitness = []
    for i, genome in enumerate(population):
        mujoco.set_mjcb_control(None)
        world       = world_func()
        world.spawn(robot_core_func().spec, spawn_position=[0, 0, 0])
        model       = world.spec.compile()
        data        = mujoco.MjData(model)

        geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

        history = []
        mujoco.set_mjcb_control(lambda m,d: nn_control(m, d, to_track, genome, history, input_size, hidden_size, output_size))

        while data.time < time:
            mujoco.mj_step(model, data)

        pos_data = np.array(history)
        distance_to_goal = xy_displacement((pos_data[-1, 0], pos_data[-1, 1]), (0.0, -0.3))
            
        results_fitness.append(distance_to_goal)
        
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


def main(robot_core_func, 
         world_func, 
         spawn_pos      = [-0.8, 0, 0.1], 
         time           = 30, 
         population     = 10, 
         generations    = 50, 
         p_crossover    = 0.5, 
         m_rate         = 0.1):

    mujoco.set_mjcb_control(None)
    world       = world_func()
    world.spawn(robot_core_func().spec, spawn_position=spawn_pos)
    model       = world.spec.compile()
    data        = mujoco.MjData(model)

    input_size  = len(data.qvel)  # e.g., 6 positions + 6 velocities  
    hidden_size = 16
    output_size = model.nu

    population  = [init_genome(input_size, hidden_size, output_size) for _ in range(population)]
    population_fitness = evaluate(population, robot_core_func, world_func, time, input_size, hidden_size, output_size)

    idx = np.argmin(population_fitness) 
    x0_best = population[idx]
    f0_best = population_fitness[idx]
    
    x_best = [x0_best]
    f_best = [f0_best]

    for _ in range(generations):
        parents, parents_f             = parent_selection(population, population_fitness)
        offsprings                     = crossover(parents, p_crossover)
        
        offsprings                     = mutate(offsprings, m_rate)
        f_offspring                    = evaluate(offsprings, robot_core_func, world_func, time, input_size, hidden_size, output_size)
        population, population_fitness = survivior_selection(
            parents, parents_f, offsprings, f_offspring
        )

        idx = np.argmin(population_fitness) # !!! SWITCHED FROM .argmax() to .argmin() FOR MINIMIZATION !!!
        xi_best = population[idx]
        fi_best = population_fitness[idx]
        if fi_best < f_best[-1]: # !!! SWITCHED FROM > to < FOR MINIMIZATION !!!
            x_best.append(xi_best)
            f_best.append(fi_best)
        else:
            x_best.append(x_best[-1])
            f_best.append(f_best[-1])

            
    
    return f_best[-1]
