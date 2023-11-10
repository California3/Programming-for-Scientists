## COMP1730/6730 Project assignment

# Your ANU ID: u7100771
# Your NAME: Guangming Zeng
# I declare that this submission is my own work
# [ref: https://www.anu.edu.au/students/academic-skills/academic-integrity ]

## You should implement the following functions
## You can define new function(s) if it helps you decompose the problem
## into smaller subproblems.

def compute_position(path_to_file, time):
    # read the file and store the data in a list
    data = np.loadtxt(path_to_file, skiprows=1, usecols=[_ for _ in range(1,5+1)])

    # data[:, 0] # Mass M
    # data[:, 1] # XPOS X
    # data[:, 2] # YPOS Y
    # data[:, 3] # XVEL Vx
    # data[:, 4] # YVEL Vy

    # (Xt, Yt)
    XY_Pos = np.zeros((data.shape[0], 2))
    
    # Xt = X + Vx * t
    XY_Pos[:, 0] = data[:, 1] + data[:, 3] * time

    # Yt = Y + Vy * t
    XY_Pos[:, 1] = data[:, 2] + data[:, 4] * time

    return XY_Pos

def compute_acceleration(path_to_file, time):
    G_const = 6.67e-11
    data = np.loadtxt(path_to_file, skiprows=1, usecols=[_ for _ in range(1,5+1)])
    XY_Pos = compute_position(path_to_file, time)

    acceleration = np.zeros((data.shape[0], 2))

    for planet_current in range(data.shape[0]):
        for planet_other in range(data.shape[0]):
            if planet_other == planet_current:
                continue
            pji_x = XY_Pos[planet_other, 0] - XY_Pos[planet_current, 0] # xj - xi
            pji_y = XY_Pos[planet_other, 1] - XY_Pos[planet_current, 1] # yj - yi
            distance = np.sqrt(pji_x**2 + pji_y**2) # sqrt[(xi - xj)**2 + (yi - yj)**2]

            # a = G mi * mj / (r**2 * mi) = G mj / r**2
            acceleration_ij = (G_const * data[planet_other, 0] / distance**2)
            # ax = a * (xj - xi) / r
            acceleration[planet_current, 0] += acceleration_ij * pji_x / distance
            # ay = a * (yj - yi) / r
            acceleration[planet_current, 1] += acceleration_ij * pji_y / distance
    return acceleration


def forward_simulation(path_to_file, total_time, num_steps):
    G_const = 6.67e-11
    data = np.loadtxt(path_to_file, skiprows=1, usecols=[_ for _ in range(1,5+1)])
    if num_steps == 0:
        t_split = 0
    else:
        t_split = total_time / num_steps
    position_forward = np.zeros((num_steps, 2 * data.shape[0]))

    # For t1
    position_current = compute_position(path_to_file, t_split) # p1
    acceleration_current = compute_acceleration(path_to_file, t_split) # a1
    velocity_current = data[:,3:5] + t_split * acceleration_current  # v1 = v0 + a1 * t
    position_forward[0,:] = np.concatenate(position_current) # store p1
    
    # For t2, t3, ...
    for step in range(1, num_steps):
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
                acceleration_ij = (G_const * data[planet_other, 0] / distance**2)
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
    
    return position_forward

def max_accurate_time(path_to_file, epsilon):
    T = 1
    while(True):
        position_forward_T = forward_simulation(path_to_file, T, T)[-1]
        position_forward_1 = forward_simulation(path_to_file, T, 1)[-1]
        error = 0
        for i in range(0, len(position_forward_T), 2):
            error += np.sqrt((position_forward_T[i] - position_forward_1[i])**2 + (position_forward_T[i+1] - position_forward_1[i+1])**2)
        if error <= epsilon:
            T += 1
        break
    return T - 1


### THIS FUNCTION IS ONLY FOR COMP6730 STUDENTS ###
def orbit_time(path_to_file, object_name):
    pass

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

test_max_accurate_time()