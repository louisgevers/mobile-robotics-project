from src import model


import random
import numpy as np
import matplotlib.pyplot as plt


def update_robot(
    robot: model.Robot,
    command: model.MotorSpeed,
    sensors: model.SensorReading,
    states,
    camera,
):
    x, P = states.Kalmanfilter(
        np.array([command.left, command.right]),
        np.array([sensors.left, sensors.right]),
        np.array([robot.position.x, robot.position.y, robot.angle]),
        camera,
    )
    robot.position.x = x[0]
    robot.position.y = x[1]
    robot.angle = x[2]


def initialise(initialposition, initialangle):
    L = 150
    thetadotvar = 0.6
    speedvar = 6.15
    posvar = 1
    thetavar = 0.01
    measvar = np.diag([1, 1, 1 / L, 0.434782608 * 6.15, 0.434782608 * 6.15 / L])
    statevar = np.diag([posvar, posvar, thetavar, speedvar, speedvar / L])
    y = filter(
        np.concatenate((initialposition, [initialangle], [0, 0])),
        np.zeros((5, 5)),
        measvar,
        statevar,
        0.1,
    )
    return y


class filter:
    def __init__(self, x0, P0, Q, R, T):

        self.x = [x0]
        self.u = []
        self.P = [P0]
        self.T = T
        self.speed = []
        self.camerapos = []
        self.L = 150
        self.R = R
        self.Q = Q
        self.u_prev = np.array([0, 0])
        self.speedconv = 0.15

    def Kalmanfilter(self, u, speed, camerapos=0, camera=False):
        # x=[x,y,theta,xdot,thetadot]
        # u=[vl,vr]
        udiff = u - self.u_prev
        self.u_prev = u
        # print(udiff)
        x = np.array(self.x[-1]).T
        Sigma = np.array(self.P[-1]).T
        theta = x[2]
        A = np.array(
            [
                [1, 0, 0, np.sin(theta) * self.T, 0],
                [0, 1, 0, np.cos(theta) * self.T, 0],
                [0, 0, 1, 0, self.T / self.L],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )

        B = np.zeros((5, 2))
        B[3:] = [[0.5, 0.5], [-1 / self.L, 1 / self.L]]
        B = self.speedconv * B
        # print(B)
        x_pred = A @ x + B @ udiff
        # print(A@x)
        # print(x_pred)
        G = np.array(
            [
                [1, 0, -self.T * np.sin(theta) * x[3], np.cos(theta) * self.T, 0],
                [0, 1, self.T * np.cos(theta) * x[4], np.sin(theta) * self.T, 0],
                [0, 0, 1, 0, self.T / self.L],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        # print(G)
        # print(Sigma)
        P_new = G @ Sigma @ G.T + self.R
        P_est = P_new
        # print(P_new)
        if camera:
            # print(camerapos,speed)
            measurement = np.append(camerapos, speed)
            M = np.eye(5)
            M[3:, 3:] = self.speedconv * np.array(
                [[0.5, 0.5], [-1 / self.L, 1 / self.L]]
            )
            C = np.eye(5)
            Q = self.Q
        else:
            measurement = speed
            M = self.speedconv * np.array([[0.5, 0.5], [-1 / self.L, 1 / self.L]])
            C = np.concatenate((np.zeros((2, 3)), np.eye(2)), axis=1)
            Q = self.Q[3:, 3:]
        y = np.ravel(M @ measurement)
        K = P_new @ C.T @ (np.linalg.inv(C @ P_new @ C.T + Q))
        # C=np.concatenate((np.zeros((2,3)),np.eye(2)),axis=1)

        x_est = x_pred + K @ (y - C @ x_pred)

        # print(y)
        # print(C)
        # print(C@x)
        a = K @ C
        P_est = (np.eye(np.shape(a)[0]) - K @ C) @ P_new

        self.x.append(x_est.tolist())
        self.P.append(P_est.tolist())
        return x_est, P_est
