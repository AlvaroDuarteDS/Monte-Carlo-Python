# This is a sample Python script.
import matplotlib.pyplot as plt
import random
import math

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and setting
landmarks = [[26.0, 4.0], [79.0, 18.0], [28.0, 97.0],
             [24.0, 74.0], [49.0, 63.0], [62.0, 47.0],
             [17.0, 33.0], [35.0, 17.0]]
world_size = 100.0
"""
Random Generators
random_device rd;
mt19937 gen(rd());
"""


def gen_real_random():
    return random.random()


def mod(first, second):
    return first - second * math.floor(first / second)


class Robot:
    def __init__(self):
        self.x = gen_real_random() * world_size
        self.y = gen_real_random() * world_size
        self.orient = gen_real_random() * 2.0 * math.pi
        self.foward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

    def set(self, new_x, new_y, new_orient):
        if new_x < 0 or new_x >= world_size:
            raise Exception("X coordinate out of bound")
        if new_y < 0 or new_y >= world_size:
            raise Exception("Y coordinate out of bound")
        if new_orient < 0 or new_orient >= 2 * math.pi:
            raise Exception("Orientation must be in [0..2pi]")
        self.x = new_x
        self.y = new_y
        self.orient = new_orient

    def set_noise(self, new_forward_noise, new_turn_noise, new_sense_noise):
        self.forward_noise = new_forward_noise
        self.turn_noise = new_turn_noise
        self.sense_noise = new_sense_noise

    def gen_gauss_random(self, mean, variance):
        return random.gauss(mean, variance)

    def gaussian(self, mu, sigma, x):
        try:
            result = math.exp(- pow(x - mu, 2) / (2 * pow(sigma, 2)))
        except ZeroDivisionError:
            result = 0
        return result

    def sense(self):
        z = []
        for i in range(0, int(len(landmarks)) ):
            dist = math.sqrt(math.pow(self.x - landmarks[i][0], 2) + math.pow(self.y - landmarks[i][1], 2))
            dist = dist + self.gen_gauss_random(0.0, self.sense_noise)
            """Aqui deberia ir z[i] = dist"""
            z.append(dist)
        return z

    def move(self, turn, forward):
        if forward < 0:
            raise Exception("Robot cannot move backward")

        """turn, and add randomness to the turning command"""
        self.orient = self.orient + turn + self.gen_gauss_random(0.0, self.turn_noise)
        self.orient = mod(self.orient, 2 * math.pi)
        """move, and add randomness to the motion command"""
        dist = forward + self.gen_gauss_random(0.0, self.foward_noise)
        self.x = self.x + (math.cos(self.orient) * dist)
        self.y = self.y + (math.sin(self.orient) * dist)
        """cyclic truncate"""
        self.x = mod(self.x, world_size)
        self.y = mod(self.y, world_size)
        """set particle"""
        res = Robot()
        res.set(self.x, self.y, self.orient)
        res.set_noise(self.foward_noise, self.turn_noise, self.sense_noise)
        return res

    def show_pose(self):
        return print("[x=", self.x, " y=", self.y, " orient=", self.orient, "]")

    def read_sensors(self):
        z = self.sense()
        readings = "["
        for i in range(0, len(z)):
            if i == len(z) - 1:
                readings = readings + str(z[i])
            else:
                readings = readings + str(z[i]) + ", "

        readings = readings + "]"
        return readings

    def measurement_prob(self, measurement):
        prob = 1.0

        for i in range(0, int(len(landmarks))):
            dist = math.sqrt(pow((self.x - landmarks[i][0]), 2) + pow((self.y - landmarks[i][1]), 2))
            prob = prob * self.gaussian(dist, self.sense_noise, measurement[i])
            # print("Probabilidad de la iteracion ", i, " = ", prob)

        return prob


def evaluation(r, p, n):
    sum = 0.0
    for i in range(0, n):
        dx = mod((p[i].x - r.x + (world_size / 2.0)), world_size) - (world_size / 2.0)
        dy = mod((p[i].y - r.y + (world_size / 2.0)), world_size) - (world_size / 2.0)
        err = math.sqrt(pow(dx, 2) + pow(dy, 2))
        sum = sum + err

    return sum / n


def max(arr, n):
    max = 0
    for i in range(0, n):
        if (arr[i] > max):
            max = arr[i]

    return max

def visualization(n, robot, step, p, pr):
    """Graph Format"""
    plt.title('MCL, step '+ str(step))
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    """Draw particles in green"""
    for i in range(0, n):
        plt.plot(p[i].x, p[i].y, "go")
    """Draw resampled particles in yellow"""
    for i in range(0, n):
        plt.plot(pr[i].x, pr[i].y, "yo")
    """Draw landmarks in red"""
    for i in range(0, int(len(landmarks))):
        plt.plot(landmarks[i][0], landmarks[i][1], "ro")
    """Draw robot position in blue"""
    plt.plot(robot.x, robot.y, "bo")
    """Save the image and close the plot"""
    plt.savefig('./Images/Step'+ str(step)+ '.png')
    plt.clf()


# Press the green butto   n in the gutter to run the script.
if __name__ == '__main__':
    """Practice Interfacing with Robot Class"""
    """myrobot = Robot()
    myrobot.set_noise(5.0, 0.1, 5.0)
    myrobot.set(30.0, 50.0, math.pi / 2.0)
    myrobot.show_pose()
    print(myrobot.read_sensors())
    print("------------------------------")
    print("The robot moves...")
    myrobot.move(-math.pi / 2.0, 15.0)
    print("------------------------------")
    myrobot.show_pose()
    print(myrobot.read_sensors())
    print("------------------------------")
    print("The robot moves...")
    myrobot.move(-math.pi / 2.0, 10.0)
    print("------------------------------")
    myrobot.show_pose()
    print(myrobot.read_sensors())"""

    n = 1000
    """Create set of particles"""
    p = [Robot() for i in range(n)]
    p4 = [Robot() for i in range(n)]
    for i in range(0, n):
        p[i].set_noise(0.05, 0.05, 5.0)

    print("-------------------------------")
    myrobot = Robot()
    myrobot.show_pose()
    """Initialize two-dimensional array for measurement"""
    z = [[0] * 3 for i in range(3)]
    steps = 50
    """Iterating 50 times over the set of particles"""
    for t in range(0, steps):
        p4.clear()
        for i in range(0, n):
            p4.append(p[i])
        """Move the robot and sense the environment afterwards"""
        myrobot = myrobot.move(0.1, 5.0)
        z = myrobot.sense()
        """Simulate a robot motion for each of these particles"""
        p2 = [Robot() for i in range(n)]
        for i in range(0, n):
            p2[i] = p[i].move(0.1, 5.0)
            p[i] = p2[i]
        """Generate particle weights depending on robot's measurement"""
        w = []
        for i in range(0, n):
            w.append(p[i].measurement_prob(z))
        """Resample the particles with a sample probability proportional to the importance weight"""
        p3 = [Robot() for i in range(n)]
        index = int(gen_real_random() * n)
        beta = 0.0
        mw = max(w, n)
        for i in range(0, n):
            beta = beta + gen_real_random() * 2.0 * mw
            while beta > w[index]:
                beta = beta - w[index]
                index = mod((index + 1), n)
            p3[i] = p[index]
        for k in range(0, n):
            p[k] = p3[k]
        visualization(n, myrobot, t, p4, p)
        print("Step = ", t, ", Evaluation = ", evaluation(myrobot, p, n))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
