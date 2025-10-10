# Third-party libraries
import mujoco
import numpy as np
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
    # Example state: joint velocities
    qvel = data.qvel

    action = forward_nn(qvel, genome, input_size, hidden_size, output_size)
    
    delta = 0.05
    data.ctrl += action * delta
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)
    history.append(to_track[0].xpos.copy())

def evaluate(population, robot_core_func, world_func, time, input_size, hidden_size, output_size, goal):
    results_fitness = []
    for i, genome in enumerate(population):
        try:
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

            if pos_data.shape[0] < 2:
                results_fitness.append(0.0)
                continue

            start_pos_3d = pos_data[0]
            end_pos_3d = pos_data[-1]
            goal_3d = np.array(goal)

            vec_start_to_goal = goal_3d - start_pos_3d
            vec_robot_displacement = end_pos_3d - start_pos_3d

            dist_start_to_goal = np.linalg.norm(vec_start_to_goal)

            if dist_start_to_goal < 1e-6:
                progress = 0.0
            else:
                progress = np.dot(vec_robot_displacement, vec_start_to_goal) / dist_start_to_goal
            
            fitness = -progress
            results_fitness.append(fitness)
        except Exception as e:
            print(f"Error evaluating controller {i}: {e}")
            results_fitness.append(float('inf')) # Assign worst fitness on error
            
    return results_fitness

def parent_selection(x, f):
    """Original parent selection based on inverted probabilities."""
    x_parents, f_parents = [],[]
    
    # Ensure fitness values are numpy array for operations
    f = np.array(f)
    
    total_f = np.sum(f)
    s = 1e-9 # Small epsilon to prevent division by zero

    # This logic requires positive fitness values, where lower is better.
    # The transformation to achieve this is done before calling this function.
    max_probs = f / (total_f + s)
    min_probs = 1 / (max_probs + s)
    
    # Check for non-negative probabilities before normalization
    if np.any(min_probs < 0):
        # This case should not be reached if fitness is shifted correctly
        min_probs[min_probs < 0] = 0

    min_probs_normalized = min_probs / (min_probs.sum() + s)
    
    # Handle potential NaN from division issues
    if np.isnan(min_probs_normalized).any():
        min_probs_normalized = np.ones(len(x)) / len(x) # Fallback to uniform selection

    for _ in range(int(len(x)/2)):
        parent_indices = np.random.choice(len(x), size=2, replace=True, p=min_probs_normalized)
        x_parents.append([x[i] for i in parent_indices])
        f_parents.append([f[i] for i in parent_indices])

    return x_parents, f_parents

def crossover(x_parents, p_crossover):
    offspring = []
    for pair in x_parents:
        if np.random.random() < p_crossover:
            p1, p2 = pair
            crossover_point_1 = np.random.randint(0, int(len(p1)))
            crossover_point_2 = np.random.randint(crossover_point_1, len(p1))
            
            for px, py in [(p1, p2), (p2, p1)]:
                strand = px[crossover_point_1:crossover_point_2]
                new_baby = np.concatenate((py[:crossover_point_1], strand, py[crossover_point_2:]))
                offspring.append(new_baby)
        else:
            offspring.extend(pair)
    return np.array(offspring)

def mutate(x, mutation_rate):
    """Apply mutation to an individual."""
    for i in x:
        for a in range(len(i)):
            if np.random.random() < mutation_rate:
                i[a] += np.random.normal(0, 0.1) # Add small gaussian noise
    return x

def survivior_selection(x, f, x_offspring, f_offspring):
    """Select the survivors for the next generation using elitism."""
    _x = [item for sublist in x for item in sublist]
    _f = [item for sublist in f for item in sublist]
    
    combined_x = np.array(list(_x) + list(x_offspring))
    combined_f = np.array(list(_f) + list(f_offspring))
    
    sorted_indices = np.argsort(combined_f)
    
    n = len(_x)
    survivors_x = combined_x[sorted_indices[:n]]
    survivors_f = combined_f[sorted_indices[:n]]

    return list(survivors_x), list(survivors_f)

def main(robot_core_func, 
         world_func, 
         spawn_pos      = [-0.8, 0, 0.1], 
         time           = 30, 
         population     = 10, 
         generations    = 50, 
         p_crossover    = 0.5, 
         m_rate         = 0.1,
         goal           = [5.0, 0.0, 0.5]):

    mujoco.set_mjcb_control(None)
    world       = world_func()
    world.spawn(robot_core_func().spec, spawn_position=spawn_pos)
    model       = world.spec.compile()
    data        = mujoco.MjData(model)

    input_size  = len(data.qvel)
    hidden_size = 16
    output_size = model.nu

    current_population  = [init_genome(input_size, hidden_size, output_size) for _ in range(population)]
    population_fitness = evaluate(current_population, robot_core_func, world_func, time, input_size, hidden_size, output_size, goal)

    idx = np.argmin(population_fitness) 
    x0_best, f0_best = current_population[idx], population_fitness[idx]
    
    x_best, f_best = [x0_best], [f0_best]

    for gen in range(generations):
        # --- Fitness Transformation ---
        fitness_np = np.array(population_fitness)
        min_fit_for_gen = fitness_np.min()
        # Shift fitness values to be non-negative for probability calculation in parent_selection
        shifted_fitness = (fitness_np - min_fit_for_gen + 1e-9).tolist()

        parents, parents_f_shifted = parent_selection(current_population, shifted_fitness)
        
        offsprings = crossover(parents, p_crossover)
        offsprings = mutate(offsprings, m_rate)
        
        # Evaluate offspring to get their original (un-shifted) fitness
        f_offspring_original = evaluate(offsprings, robot_core_func, world_func, time, input_size, hidden_size, output_size, goal)
        
        # Shift offspring fitness using the same factor as the parents for a fair comparison
        f_offspring_shifted = (np.array(f_offspring_original) - min_fit_for_gen + 1e-9).tolist()
        
        # Survivor selection compares the shifted fitness values
        next_pop, next_pop_fitness_shifted = survivior_selection(
            parents, parents_f_shifted, offsprings, f_offspring_shifted
        )
        
        # --- Reverse Transformation ---
        # Revert the fitness of the new population back to the original scale
        current_population = next_pop
        population_fitness = (np.array(next_pop_fitness_shifted) + min_fit_for_gen - 1e-9).tolist()

        # Track the best individual using the original fitness scale
        idx = np.argmin(population_fitness)
        xi_best = current_population[idx]
        fi_best = population_fitness[idx]
        
        if fi_best < f_best[-1]:
            x_best.append(xi_best)
            f_best.append(fi_best)
        else:
            x_best.append(x_best[-1])
            f_best.append(f_best[-1])
    
    return f_best[-1]

