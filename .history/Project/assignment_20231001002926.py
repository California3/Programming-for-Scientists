## COMP1730/6730 Project assignment

# Your ANU ID: u7100771
# Your NAME: Guangming Zeng
# I declare that this submission is my own work
# [ref: https://www.anu.edu.au/students/academic-skills/academic-integrity ]

## You should implement the following functions
## You can define new function(s) if it helps you decompose the problem
## into smaller subproblems.

def get_data(path_to_file):
    """
    Reads data from a file and returns a numpy array.

    Args:
        path_to_file (str): The path to the file to read.

    Returns:
        numpy.ndarray: A numpy array containing the data from the file.

    Example:
        >>> get_data('/path/to/file.csv')
        array([[1, 2, 3, 4, 5],
               [6, 7, 8, 9, 10]])
    """
    # data[:, 0] # Mass M
    # data[:, 1] # XPOS X
    # data[:, 2] # YPOS Y
    # data[:, 3] # XVEL Vx
    # data[:, 4] # YVEL Vy
    return np.loadtxt(path_to_file, skiprows=1, usecols=[_ for _ in range(1,5+1)])

def compute_position(path_to_file, time):
    """
    Computes the position of each mass in the file at a given time.
    xt = x + vx * t
    yt = y + vy * t

    Args:
        path_to_file (str): The path to the file containing the mass data.
        time (float): The time at which to compute the position.

    Returns:
        numpy.ndarray: A 2D array containing the X and Y positions of each mass at the given time.
    """
    # read the file and store the data in a list
    data = get_data(path_to_file)

    # (xt, yt)
    xy_pos = np.zeros((data.shape[0], 2))
    
    # xt = x + vx * t
    xy_pos[:, 0] = data[:, 1] + data[:, 3] * time

    # yt = y + vy * t
    xy_pos[:, 1] = data[:, 2] + data[:, 4] * time

    return xy_pos

def compute_acceleration(path_to_file, time):
    """
    Computes the acceleration of each planet in the system at a given time.
    r = sqrt[(xi - xj)**2 + (yi - yj)**2]
    a = G mi * mj / (r**2 * mi) = G mj / r**2
    ax = a * (xj - xi) / r
    ay = a * (yj - yi) / r

    Args:
    - path_to_file (str): The path to the file containing the data for the planets.
    - time (float): The time at which to compute the acceleration.

    Returns:
    - acceleration (numpy.ndarray): A 2D numpy array of shape (n, 2), where n is the number of planets in the system.
    The i-th row of the array contains the x and y components of the acceleration of the i-th planet.
    """
    g_const = 6.67e-11
    data = get_data(path_to_file)
    xy_pos = compute_position(path_to_file, time)

    acceleration = np.zeros((data.shape[0], 2))

    for planet_current in range(data.shape[0]):
        for planet_other in range(data.shape[0]):
            if planet_other == planet_current:
                continue
            pji_x = xy_pos[planet_other, 0] - xy_pos[planet_current, 0] # xj - xi
            pji_y = xy_pos[planet_other, 1] - xy_pos[planet_current, 1] # yj - yi
            distance = np.sqrt(pji_x**2 + pji_y**2) # sqrt[(xi - xj)**2 + (yi - yj)**2]

            # a = G mi * mj / (r**2 * mi) = G mj / r**2
            acceleration_ij = (g_const * data[planet_other, 0] / distance**2)
            # ax = a * (xj - xi) / r
            acceleration[planet_current, 0] += acceleration_ij * pji_x / distance
            # ay = a * (yj - yi) / r
            acceleration[planet_current, 1] += acceleration_ij * pji_y / distance
    return acceleration

def forward_simulation(path_to_file: str, total_time: float, num_steps: int) -> np.ndarray:
    """
    Simulates the forward motion of a system using the given input file, total time, and number of steps.
    Split it into two parts. Part 1 only has 1 step. Part 2 has num_steps - 1 steps.
    Spliting is not useful for current task only. But QUIET USEFUL for tasks later. Avoid repeating calculation. Could SAVE A LOT OF TIME.

    Args:
        path_to_file (str): The path to the input file.
        total_time (float): The total time to simulate.
        num_steps (int): The number of steps to take during the simulation.

    Returns:
        np.ndarray: A numpy array containing the positions of the system at each time step.
    """
    # for t1
    position_forward, position_current, velocity_current, acceleration_current, data = forward_simulation_speedup(path_to_file, 1, total_time / num_steps)
    # for t2, t3, ...
    position_forward2 = forward_simulation_speedup_extend(position_current, velocity_current, acceleration_current, num_steps - 1, total_time / num_steps, data)[0]
    # combine t1 and t2, t3, ...
    return np.vstack((position_forward, position_forward2))

def forward_simulation_speedup(path_to_file, num_steps, t_split):
    """
    Simulates the forward motion of a system given a path to a data file, number of steps, and time split.

    Args:
        path_to_file (str): The path to the data file.
        num_steps (int): The number of steps to simulate.
        t_split (float): The time split for each step.

    Returns:
        tuple: A tuple containing:
            - position_forward (numpy.ndarray): A 2D numpy array of shape (num_steps, 2 * data.shape[0]) representing the forward positions.
            - position_current (numpy.ndarray): A 2D numpy array of shape (data.shape[0], 2) representing the current positions.
            - velocity_current (numpy.ndarray): A 2D numpy array of shape (data.shape[0], 2) representing the current velocities.
            - acceleration_current (numpy.ndarray): A 2D numpy array of shape (data.shape[0], 2) representing the current accelerations.
            - data (numpy.ndarray): A 2D numpy array of shape (num_samples, 5) representing the data from the file.
    """
    data = get_data(path_to_file)
    position_forward = np.zeros((num_steps, 2 * data.shape[0]))

    # For t1
    position_current = compute_position(path_to_file, t_split) # p1
    acceleration_current = compute_acceleration(path_to_file, t_split) # a1
    velocity_current = data[:,3:5] + t_split * acceleration_current  # v1 = v0 + a1 * t
    position_forward[0,:] = np.concatenate(position_current) # store p1

    return position_forward, position_current, velocity_current, acceleration_current, data

def forward_simulation_speedup_extend(position_current, velocity_current, acceleration_current, extend_steps, t_split, data):
    """
    Simulates the forward motion of a system of planets using numerical integration.
    
    Args:
    - position_current (numpy.ndarray): The current position of the planets in the system.
    - velocity_current (numpy.ndarray): The current velocity of the planets in the system.
    - acceleration_current (numpy.ndarray): The current acceleration of the planets in the system.
    - extend_steps (int): The number of steps to simulate forward.
    - t_split (float): The time step size.
    - data (numpy.ndarray): An array containing the mass of each planet in the system.
    
    Returns:
    - position_forward (numpy.ndarray): An array containing the positions of the planets at each step.
    - position_current (numpy.ndarray): The final position of the planets after the simulation.
    - velocity_current (numpy.ndarray): The final velocity of the planets after the simulation.
    - acceleration_current (numpy.ndarray): The final acceleration of the planets after the simulation.
    """
    g_const = 6.67e-11
    
    position_forward = np.zeros((extend_steps, 2 * data.shape[0]))
    
    # For t2, t3, ...
    for step in range(0, extend_steps):
        position_next = position_current + t_split * velocity_current # p2 = p1 + v1 * t, p3 = ...

        acceleration_next = np.zeros((data.shape[0], 2)) # a2, a3, ...
        for planet_current in range(data.shape[0]):
            for planet_other in range(data.shape[0]):
                if planet_other == planet_current:
                    continue
                pji_x = position_next[planet_other, 0] - position_next[planet_current, 0] # xj - xi
                pji_y = position_next[planet_other, 1] - position_next[planet_current, 1] # yj - yi
                distance = np.sqrt(pji_x**2 + pji_y**2) # sqrt[(xi - xj)**2 + (yi - yj)**2]

                # a = G mi * mj / (r**2 * mi) = G mj / r**2
                acceleration_ij = (g_const * data[planet_other, 0] / distance**2)
                # ax = a * (xj - xi) / r
                acceleration_next[planet_current, 0] += acceleration_ij * pji_x / distance
                # ay = a * (yj - yi) / r
                acceleration_next[planet_current, 1] += acceleration_ij * pji_y / distance

        velocity_next = velocity_current + t_split * acceleration_next # v2 = v1 + a2 * t, v3 = ...

        position_forward[step,:] = np.concatenate(position_next) # store p2, p3, ...

        # update for next step
        position_current = position_next
        velocity_current = velocity_next
        acceleration_current = acceleration_next
    
    return position_forward, position_current, velocity_current, acceleration_current

def max_accurate_time(path_to_file: str, epsilon: float) -> int:
    """
    This function calculates the maximum time T such that the Euclidean distance between the "good" configuration at time T and the "approximate" configuration at time T is less than epsilon.

    Args:
    - path_to_file (str): the path to the file containing the data for the simulation
    - epsilon (float): the maximum allowable error between the "good" and "approximate" configurations

    Returns:
    - int: the maximum time T such that the Euclidean distance between the "good" configuration at time T and the "approximate" configuration at time T is less than epsilon
    """

    # special case
    if epsilon < 0:
        return 0
    
    # For cache control. I don't want to repeat the calculation.
    position_forward_T_1, position_current, velocity_current, acceleration_current, data = forward_simulation_speedup(path_to_file, 1, 1)
    T_cal_max = 10
    position_forward_T_x, position_current, velocity_current, acceleration_current = forward_simulation_speedup_extend(position_current, velocity_current, acceleration_current, T_cal_max - 1, 1, data)
    position_forward_T_cal = np.vstack((position_forward_T_1, position_forward_T_x))

    # Find T using the binary search algorithm
    T = 1
    last_over = None
    last_not_over = None
    while(True):
        # obtain a “good” configuration at time T
        # Also for cache control. I don't want to repeat the calculation. 
        # If cache is enough, get from cache. If not, generate it and add to cache.
        if T <= T_cal_max:
            position_forward_T = position_forward_T_cal[:T, :][-1]
        else:
            position_forward_T_x2, position_current, velocity_current, acceleration_current = forward_simulation_speedup_extend(position_current, velocity_current, acceleration_current, T  - T_cal_max, 1, data)
            position_forward_T_cal = np.vstack((position_forward_T_cal, position_forward_T_x2))
            position_forward_T = position_forward_T_cal[:T, :][-1]
            T_cal_max = T

        # another “approximate” configuration
        position_forward_1 = forward_simulation(path_to_file, T, 1)[-1]

        # compute the error between the two configurations using the Euclidean distance
        error = 0
        for i in range(0, len(position_forward_T), 2):
            error += np.sqrt((position_forward_T[i] - position_forward_1[i])**2 + (position_forward_T[i+1] - position_forward_1[i+1])**2)

        # apply binary search algorithm to find the largest T such that the error is less than epsilon
        if error <= epsilon:
            last_not_over = T
            if last_over is None:
                T = T * 2
            elif last_over - last_not_over <= 1:
                return last_not_over
            else:
                T = int((T + last_over) / 2)
        else:
            last_over = T
            T = int((T + last_not_over) / 2)


### THIS FUNCTION IS ONLY FOR COMP6730 STUDENTS ###
def orbit_time(path_to_file, object_name):
    """
    This function calculates the time it takes for an object to complete one orbit around a central body.
    It takes in the path to a file containing data about the objects in the system and the name of the object we want to calculate the orbit time for.
    The function uses a range of epsilons to find the most accurate time step to simulate the orbit.
    It then uses a while loop to simulate the orbit and adjust the step size until it finds the point where the object completes one orbit.
    The function returns the time it takes for the object to complete one orbit.
    """
    # read the file and store the data in a list
    names_np = np.genfromtxt(path_to_file, skip_header=1, usecols=0, dtype=str)
    object_index = np.where(names_np == object_name)[0][0]

    data = get_data(path_to_file)
    object_position = data[object_index, 1:3]
    
    # Initialize the range of epsilons that we consider.
    epsilons = [1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000,1000000, 12500000, 15000000, 20000000, 25000000, 50000000, # all solar_system.tsv should be able to stop at this line
                100000000, 1000000000, 10000000000, 100000000000, 1000000000000,10000000000000, 100000000000000, 1000000000000000, # Something that more terrible may stop at this line
                10000000000000000]
    for epsilon in epsilons:
        # get prefer Time-Step with epsilon
        T_step = max_accurate_time(path_to_file, epsilon)

        # Initialize, record the distance and last distance,
        # Also, record the tendency of distance change, including last time.
        last_distance = None
        last_change_in_distance = None
        change_in_distance = None
        # Initialize, how to increase the step each time in while loop
        step_by_step = 2500 * 2
        # Flag, re-execute the current while loop if true.
        restart = False
        # Record the actual iteration number
        iter_cnt = 0

        # Initialize, get the first step for T_step. Num of steps is 1.
        position_forward_T_1, position_current, velocity_current, acceleration_current, _ = forward_simulation_speedup(path_to_file, 1, T_step)
        step_max = 10
        # Add to cache, get the first 10 steps for T_step. Num of steps is 10. We have executed 1 step before, so we need 9 more steps.
        position_forward_T_x, position_current, velocity_current, acceleration_current = forward_simulation_speedup_extend(position_current, velocity_current, acceleration_current, step_max - 1, T_step, data)
        # Cache initialization is done. We have 10 steps in cache now.
        position_forward_T_cal = np.vstack((position_forward_T_1, position_forward_T_x))

        step = 1000

        while(True):
            # Count the number of iterations.
            iter_cnt += 1

            # Reset the flag. 
            restart = False

            # If step per loop is too small, we can stop the loop. And we can get the result which is acceptable. Terminate this function.
            if step_by_step < 25:
                print("\n \033[42m",object_name,": Finish! T: ", T, "step: ", step, "step_by_step: ", step_by_step, "T_step: ", T_step, "epsilon: ", epsilon, "\033[0m \n")
                return T
            # If step per loop is too large or run too many iterations, give up current epsilon. Try another epsilon / Time-Step.
            if step_by_step >= 40000 or iter_cnt > 50:
                print("--------------- epsilon Upgrades. Change Step_Time.--------------- ")
                break

            # get actual Time for whole steps
            T = T_step * step


            # get object position after "step" time steps.
            # get from cache if possible.
            if step <= step_max:
                object_position_after = position_forward_T_cal[:step, object_index*2:object_index*2+2][-1]
            # generate it and add to cache if not. We have "step_max" steps in cache now.
            else:
                position_forward_T_x2, position_current, velocity_current, acceleration_current = forward_simulation_speedup_extend(position_current, velocity_current, acceleration_current, step - step_max, T_step, data)
                position_forward_T_cal = np.vstack((position_forward_T_cal, position_forward_T_x2))
                object_position_after = position_forward_T_cal[:step, object_index*2:object_index*2+2][-1]
                step_max = step

            # get distance between object init position and object position after "step" time steps.
            distance = np.sqrt((object_position_after[0] - object_position[0])**2 + (object_position_after[1] - object_position[1])**2)

            if last_distance is not None:
                # Record the tendency of distance change for current loop.
                change_in_distance = distance - last_distance
                # If history tendency is negative and current tendency is positive, make step per loop smaller and restart current loop.
                # Slow down the step.
                # Because we have passed the point we want to find.
                if last_change_in_distance is not None and last_change_in_distance <= 0 and change_in_distance >= 0:
                    step = step - step_by_step
                    step_by_step = int(step_by_step / 2)
                    step = step + step_by_step
                    restart = True
                    iter_cnt -= 1
                    print("--------------- step_by_step Downgrades --------------- ")
                # If history tendency is positive and current tendency is even larger, make step per loop larger and restart current loop.
                # Speed up the step. 
                # Still too far away from the point we want to find.
                elif last_change_in_distance is not None and last_change_in_distance > 0 and change_in_distance >= last_change_in_distance:
                    step = step - step_by_step
                    step_by_step = int(step_by_step * 2)
                    step = step + step_by_step
                    restart = True
                    iter_cnt -= 1
                    print("--------------- step_by_step Upgrades! --------------- ")
                # default action, record some data for next loop.
                elif not restart:
                    # Record the tendency of distance change which is the last time for next loop. (History tendency)
                    last_change_in_distance = change_in_distance
            # default action, record some data for next loop.
            if not restart:
                last_distance = distance
                step = step + step_by_step
            print("Iter:", iter_cnt,"Current T:", T, "Current step:", step, "step_by_step:", step_by_step, "epsilons:", epsilon)
    # Never reach here if set as many epsilons as possible.
    print("Timeout! ")
    return float('inf')
    

################################################################################
#                  VISUALISATION
################################################################################

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import numpy as np

def plot_configuration(xpositions, ypositions, object_names = []):
    """
        Plot the planetary system
        xpositions: sequence of x-coordinates of all objects
        ypositions: sequence of y-coordinates of all objects
        object_names: (optional) names of the objects
    """

    fig, ax = plt.subplots()
    
    marker_list = ['o', 'X', 's']
    color_list = ['r', 'b', 'y', 'm']
    if len(object_names) == 0:
        object_names = list(range(len(xpositions)))

    for i, label in enumerate(object_names):
          ax.scatter(xpositions[i], 
                     ypositions[i], 
                     c=color_list[i%len(color_list)],
                     marker=marker_list[i%len(marker_list)], 
                     label=object_names[i],
                     s=70)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.grid()

    plt.xlabel("x-coordinate (meters)")
    plt.ylabel("y-coordinate (meters)")
    plt.tight_layout()
    plt.show()

def visualize_forward_simulation(table, num_steps_to_plot, object_names = []):
    """
        visualize the results from forward_simulation
        table: returned value from calling forward_simulation
        num_steps_to_plot: number of selected rows from table to plot
        object_names: (optional) names of the objects
    """
    table = np.array(table)
    if len(object_names) == 0:
        object_names = list(range(len(table[0])//2))

    assert len(object_names)==len(table[0])//2

    fig = plt.figure()
    num_objects = len(table[0])//2
    xmin = min(table[:,0])
    xmax = max(table[:,0])
    ymin = min(table[:,1])
    ymax = max(table[:,1])
    for i in range(1, num_objects):
        xmin = min(xmin, min(table[:,i*2]))
        xmax = max(xmax, max(table[:,i*2]))
        ymin = min(ymin, min(table[:,(i*2+1)]))
        ymax = max(ymax, max(table[:,(i*2+1)]))

    ax = plt.axes(xlim=(xmin, 1.2*xmax), ylim=(ymin, 1.2*ymax))

    k=len(table[0])//2

    lines=[]
    for j in range(1,k): # Here we are assuming that the first object is the star
       line, = ax.plot([], [], lw=2, label=object_names[j])
       line.set_data([], [])
       lines.append(line)

    N=len(table)
    def animate(i):
        print(i)
        step_increment=N//num_steps_to_plot
        for j in range(1,k): # Here we are assuming that the first object is the star
           leading_object_trajectories=table[0:i*step_increment]
           x = [ ts[2*j] for ts in leading_object_trajectories ]
           y = [ ts[2*j+1] for ts in leading_object_trajectories ]
           lines[j-1].set_data(x, y)
        return lines
    
    fig.legend()
    plt.grid()
    matplotlib.rcParams['animation.embed_limit'] = 1024
    anim = FuncAnimation(fig, animate, frames=num_steps_to_plot, interval=20, blit=False)
    plt.show()
    return anim

## Un-comment the lines below to show an animation of 
## the planets in the solar system during the next 100 years, 
## using 10000 time steps, with only 200 equispaced time steps 
## out of these 10000 steps in the whole [0,T] time interval 
## actually plotted on the animation
#object_trajectories=forward_simulation("solar_system.tsv", 31536000.0*100, 10000)
#animation=visualize_forward_simulation(object_trajectories, 200)

## Un-comment the lines below to show an animation of 
## the planets in the TRAPPIST-1 system during the next 20 DAYS, 
## using 10000 time steps, with only 200 equispaced time steps 
## out of these 10000 steps in the whole [0,T] time interval 
## actually plotted on the animation
#object_trajectories=forward_simulation("trappist-1.tsv", 86400.0*20, 10000)
#animation=visualize_forward_simulation(object_trajectories, 200)

################################################################################
#               DO NOT MODIFY ANYTHING BELOW THIS POINT
################################################################################    

def test_compute_position():
    '''
    Run tests of the forward_simulation function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    position = np.array(compute_position("solar_system.tsv", 86400))
    truth = np.array([[ 0.00000000e+00,  0.00000000e+00],
       [ 4.26808000e+10,  2.39780800e+10],
       [-1.01015040e+11, -3.75684800e+10],
       [-1.13358400e+11, -9.94612800e+10],
       [ 3.08513600e+10, -2.11534304e+11],
       [ 2.11071360e+11, -7.44638848e+11],
       [ 6.54704160e+11, -1.34963798e+12],
       [ 2.37964662e+12,  1.76044582e+12],
       [ 4.39009072e+12, -8.94536896e+11]])
    assert len(position) == len(truth)
    for i in range(0, len(truth)):
        assert len(position[i]) == len(truth[i])
        if np.linalg.norm(truth[i]) == 0.0:
            assert np.linalg.norm(position[i] - truth[i]) < 1e-6
        else:    
            assert np.linalg.norm(position[i] - truth[i])/np.linalg.norm(truth[i]) < 1e-6
    print("all tests passed")
    
def test_compute_acceleration():
    '''
    Run tests of the compute_acceleration function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    acceleration = np.array(compute_acceleration("solar_system.tsv", 10000))
    truth = np.array([[ 3.42201832e-08, -2.36034356e-07],
       [-4.98530431e-02, -2.26599078e-02],
       [ 1.08133552e-02,  3.71768441e-03],
       [ 4.44649916e-03,  3.78461924e-03],
       [-3.92422837e-04,  2.87361538e-03],
       [-6.01036812e-05,  2.13176213e-04],
       [-2.58529454e-05,  5.32663462e-05],
       [-1.21886258e-05, -9.01929841e-06],
       [-6.48945783e-06,  1.32120968e-06]])
    assert len(acceleration) == len(truth)
    for i in range(0, len(truth)):
        assert len(acceleration[i]) == len(truth[i])
        assert np.linalg.norm(acceleration[i] - truth[i])/np.linalg.norm(truth[i]) < 1e-6
    print("all tests passed")

def test_forward_simulation():
    '''
    Run tests of the forward_simulation function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    trajectories = forward_simulation("solar_system.tsv", 31536000.0*100, 10)

    known_position = np.array([[ 1.63009260e+10,  4.93545018e+09],
       [-8.79733713e+13,  1.48391575e+14],
       [ 3.54181417e+13, -1.03443654e+14],
       [ 5.85930535e+13, -7.01963073e+13],
       [ 7.59849728e+13,  1.62880599e+13],
       [ 2.89839690e+13,  1.05111979e+13],
       [ 3.94485026e+12,  6.29896920e+12],
       [ 2.84544375e+12, -3.06657485e+11],
       [-4.35962396e+12,  2.04187940e+12]])
   
    rtol = 1.0e-6
    last_position = np.array(trajectories[-1]).reshape((-1,2))
    for j in range(len(last_position)):
        x=last_position[j]
        assert np.linalg.norm(x-known_position[j])/np.linalg.norm(known_position[j]) < rtol, "Test Failed!"

    print("all tests passed")

def test_max_accurate_time():
    '''
    Run tests of the max_accurate_time function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    assert max_accurate_time('solar_system.tsv', 1) == 5.0
    assert max_accurate_time('solar_system.tsv', 1000) == 163.0
    assert max_accurate_time('solar_system.tsv', 100000) == 1632.0
    print("all tests passed")

def test_orbit_time():
    '''
    Run tests of the orbit_time function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''

    # accepting error of up to 10%
    assert abs(orbit_time('solar_system.tsv', 'Mercury')/7211935.0 - 1.0) < 0.1
    assert abs(orbit_time('solar_system.tsv', 'Venus')/19287953.0 - 1.0) < 0.1
    assert abs(orbit_time('solar_system.tsv', 'Earth')/31697469.0 - 1.0) < 0.1
    assert abs(orbit_time('solar_system.tsv', 'Mars')/57832248.0 - 1.0) < 0.1
    print("all tests passed")

test_compute_position()
test_compute_acceleration()
test_forward_simulation()
test_max_accurate_time()
test_orbit_time()

# orbit_time('solar_system.tsv', 'Jupiter')
# orbit_time('solar_system.tsv', 'Saturn')
# orbit_time('solar_system.tsv', 'Uranus')
# orbit_time('solar_system.tsv', 'Neptune')