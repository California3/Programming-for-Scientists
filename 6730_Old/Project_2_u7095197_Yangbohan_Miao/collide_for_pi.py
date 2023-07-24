import decimal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

pi = '3.14159265358979323846'


def clac_clazz(v1, v2, x1, x2):
    '''
    Ball-Ball collision: 1
    Ball-Wall collision: 2
    Stop: 0
    Finish: -1
    '''

    _0 = decimal.Decimal(0)
    if _0 < v1 < v2:
        return 0
    elif _0 < v2 < v1:
        return 1
    elif v2 < 0 < v1:
        return 1
    elif v1 < _0 < v2:
        return 2
    elif v1 < v2 < _0:
        return 2
    elif v2 < v1 < _0:
        if x1 / v1 < x2 / v2:
            return 1
        elif x1 / v1 > x2 / v2:
            return 2
        else:
            # Reach the wall at the same time.
            return -1
    else:
        return 1


def next_collide(m1, m2, v1, v2, x1, x2, clazz):
    if clazz == 1:
        v1_next = ((m1 - m2) * v1) / (m1 + m2) + (decimal.Decimal(2) * m2 * v2) / (m1 + m2)
        v2_next = (decimal.Decimal(2) * m1 * v1) / (m1 + m2) - ((m1 - m2) * v2) / (m1 + m2)
        duration = abs((x2 - x1) / (v2_next - v1_next))
        x1_next = x1 + v1 * duration
        x2_next = x2 + v2 * duration

    else:
        v1_next = -v1
        v2_next = v2
        duration = abs(x1 / v1)
        x1_next = x1 + v1 * duration
        x2_next = x2 + v2 * duration

    return m1, m2, v1_next, v2_next, x1_next, x2_next, duration


def play(statuses):
    def update(frame):
        collides = []
        time = decimal.Decimal(frame) / 10
        for i in range(len(statuses)):
            goal_time = statuses[i][6]
            if time < goal_time:
                collides = [statuses[i - 1], statuses[i]]
                break

        if len(collides) == 0:
            ax.set_title(f"{len(statuses) - 1} collisions have been completed.")
            ani.event_source.stop()
            return

        ax.clear()
        ax.set_xlim(0, 10)

        ball_1 = collides[0][4] + (collides[1][4] - collides[0][4]) * (time - collides[0][6]) / (
                    collides[1][6] - collides[0][6])
        ball_2 = collides[0][5] + (collides[1][5] - collides[0][5]) * (time - collides[0][6]) / (
                    collides[1][6] - collides[0][6])

        ax.plot(ball_2, 0, 'bo', markersize=15)
        ax.plot(ball_1, 0, 'ro', markersize=10)

        ax.set_title(f"{i - 1} collisions have been completed.")

    fig, ax = plt.subplots(figsize=(6, 3))
    ani = FuncAnimation(fig, update, interval=100, repeat=False)
    plt.show()


if __name__ == '__main__':
    # Define the precision of pi.
    N = decimal.Decimal(5)

    # Define the basic parameters.
    m1 = decimal.Decimal(1)
    m2 = decimal.Decimal(m1 * (10 ** (2 * N)))
    v1 = decimal.Decimal(0)
    v2 = decimal.Decimal(-1)
    x1 = decimal.Decimal(1)
    x2 = decimal.Decimal(2)

    # Define the system.
    count = decimal.Decimal(0)
    goal_time = decimal.Decimal(0)
    statuses = []
    statuses.append([m1, m2, v1, v2, x1, x2, goal_time, count])

    # Start the Collision.
    while True:
        # One time collision.
        clazz = clac_clazz(v1, v2, x1, x2)
        if clazz == -1:
            print(f'Error!')
            quit()
        elif clazz == 0:
            print(f'Finish!')
            break
        else:
            m1, m2, v1, v2, x1, x2, duration = next_collide(m1, m2, v1, v2, x1, x2, clazz)
            count += decimal.Decimal(1)
            goal_time += duration
            statuses.append([m1, m2, v1, v2, x1, x2, goal_time, count])

    # Test the result.
    print(f'N = {N}, count = {count}, pi = {pi[:int(N) + 2]}')
    if str(count) == pi[:int(N) + 2].replace('.', ''):
        print(f'Pass The Test!')
    else:
        print(f'Fail The Test!')

    # Play the animation.
    play(statuses)
