我的项目就是通过模拟这些碰撞来计算出总的碰撞次数。

首先我是对于两个球在运动过程中的状态进行分析，计算出两个小球在运动过程中的发生碰撞的时间点以及碰撞后状态。然后将小球的状态用动画的形式去演示出来。

This project aims to calculate the value of $\pi$ through a simulation of perfectly elastic collisions between two balls. The project is divided into two parts: the calculation of collisions and the visualization of the simulation. The code is written in Python using several standard libraries and packages, including `Decimal` from the `decimal` module, and `FuncAnimation` from the `matplotlib.animation` module.





在预处理部分，由于小球的碰撞是在瞬间发生的，因此我导入了decimal库用于进行高精度计算。

In the preprocessing section, as the collision of small balls occurs instantaneously, I imported the decimal library for high-precision calculations.

第二个难点在于，需要同时计算小球的速度与位移来判断小球是发生小球与小球的碰撞还是小球与墙壁的碰撞。我先将这个动画中所有的碰撞情况全部计算出来，并保存在一个碰撞列表中。这个列表中有两个小球的每次碰撞的时间，位移，速度。这些数据用于计算动画过程中小球的运动状态。

The second difficulty lies in the need to simultaneously calculate the velocity and displacement of the ball to determine whether the ball has collided with the ball or with the wall. I will first calculate all the collision situations in this animation and save them in a collision list. This list includes the time, displacement, and velocity of each collision between two small balls. These data are used to calculate the motion state of the ball during the animation process.

还有一个难点就在于当两个小球接近墙壁时，会发生非常多次的碰撞。这些碰撞发生的时间非常短，而由于动画是一帧一帧进行的，因此可能在每一帧的变化中错过了小球的碰撞时间点。但是我们仍需要对于小球的运动状态进行正确的展现，即可能有些碰撞不在动画演示中，但是每一帧的动画必须实时展现小球的位置与速度。

Another difficulty is that when two small balls approach the wall, there will be very many collisions. These collisions occur in a very short amount of time, and since the animation is frame by frame, it is possible to miss the collision time point of the ball in each frame of change. However, we still need to present the motion state of the ball correctly, that is, some collisions may not be included in the animation demonstration, but each frame of the animation must display the position and speed of the ball in real-time.



list的问题：





尝试：

I previously attempted to use time intervals to calculate the state and number of collisions, and then demonstrated the motion status of two small balls in real-time through animation. But I found that as n increased, the number of collisions between the two small balls would significantly increase, leading to the loss of a lot of collision information. Therefore, I decided to improve my program by first calculating all the collision situations before simulating the animation.





细节：

利用当前时间减去上次碰撞时间占两次碰撞间隔时间的百分比，我就可以计算出动画帧出现时两个小球所在的位置。这样就能模拟出小球运动的效果了。

By subtracting the percentage of the last collision time from the current time, I can calculate the position of the two small balls when the animation frame appears. This will simulate the effect of ball movement.



对于程序，我采用循环体一直不停判断两个小球的运动状态，并将小球的碰撞分成球与球碰撞以及球与墙碰撞全部记录在list中。当计算完所有碰撞信息后，再对动画进行演示。

For the program, I use a loop to continuously determine the motion status of two small balls, and record the collisions of the balls into ball to ball collisions and ball to wall collisions in the list. After calculating all collision information, demonstrate the animation.

 The program uses lists to store the status of the system during the simulation, and use `Decimal` to store quality speed and position.



这一部分我是讨论了两个小球运动中所有可能出现的情况，并出现的碰撞情况：1代表小球间的碰撞，2代表球与墙壁间的碰撞，0代表着两个小球同时向右且左边的小球速度没有右边的快，即整个碰撞过程全部结束。

In this section, I discussed all possible situations in the motion of two small balls, and the collision situations that may occur: 1 represents the collision between the balls, 2 represents the collision between the balls and the wall, and 0 represents the simultaneous rightward movement of two small balls and the left ball's velocity is not as fast as the right ball, indicating that the entire collision process is over.



这一部分是计算两种碰撞的情况，分别是球与墙壁的碰撞以及球与球的碰撞。然后将所有碰撞的数据全部存起来，用于计算下次的碰撞。

This section calculates two collision scenarios, namely the collision between a ball and a wall, and the collision between a ball and a ball. Then store all the collision data for calculating the next collision.



这一部分是处理动画的每一帧中小球的动画。根据相邻两次碰撞计算出的小球在两次碰撞时的位置，再通过动画帧的时间计算出动画帧中，小球所在的位置。

This section deals with the animation of the small ball in each frame of the animation. Calculate the position of the small ball during two adjacent collisions, and then calculate the position of the small ball in the animation frame based on the time of the animation frame.



test：

The test data was created by specifying the initial conditions and desired precision for $\pi$. The program then simulates the collisions and compares the calculated value with the known value of $\pi$. The program will report the test results, that is, whether it is equal to the actual $\pi$.



I tried testing with different data and the results were all displayed correctly. However, for larger n, the program's running time will become very slow, and the animation will also appear relatively sluggish.