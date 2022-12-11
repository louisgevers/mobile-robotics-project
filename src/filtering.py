from src import model
import numpy as np


class Filter:
    def __init__(
        self,
        initial_state: model.Robot,
    ) -> None:
        # Constants
        self.L = 150
        self.T = 0.1
        SPEED_VAR = 6.15
        SPEED_CONVERSION = 0.15
        POS_VAR = 1
        THETA_VAR = 0.01

        # Initialize variables for kalman filter
        self.x = np.concatenate(
            (initial_state.position.v, [initial_state.angle], [0, 0])
        )
        self.u = []
        self.P = np.zeros((5, 5))
        self.speed = []
        self.camerapos = []
        self.R = np.diag(
            [
                POS_VAR,
                POS_VAR,
                THETA_VAR,
                SPEED_VAR,
                SPEED_VAR / self.L,
            ]
        )
        self.Q = np.diag(
            [
                1,
                1,
                1 / self.L,
                SPEED_CONVERSION * SPEED_VAR,
                SPEED_CONVERSION * SPEED_VAR / self.L,
            ]
        )
        self.u_prev = np.array([0, 0])
        self.speedconv = 0.15

    def update_robot(
        self,
        robot: model.Robot,
        command: model.MotorSpeed,
        sensors: model.SensorReading,
        camera: bool = True,
    ) -> model.Robot:
        x = self.__kalman_filter(
            np.array([command.left, command.right]).ravel(),
            np.array([sensors.motor.left, sensors.motor.right]).ravel(),
            np.array([robot.position.x, robot.position.y, robot.angle]).ravel(),
            camera,
        )
        return model.Robot(position=model.Point(x=x[0], y=x[1]), angle=x[2])

    def __kalman_filter(
        self,
        u: np.ndarray,
        speed: np.ndarray,
        camerapos: np.ndarray,
        camera: bool = False,
    ) -> np.ndarray:
        udiff = u - self.u_prev
        self.u_prev = u

        x = self.x.T
        Sigma = self.P.T
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

        x_pred = A @ x + B @ udiff

        G = np.array(
            [
                [1, 0, -self.T * np.sin(theta) * x[3], np.cos(theta) * self.T, 0],
                [0, 1, self.T * np.cos(theta) * x[4], np.sin(theta) * self.T, 0],
                [0, 0, 1, 0, self.T / self.L],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )

        P_new = G @ Sigma @ G.T + self.R
        P_est = P_new

        if camera:
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

        y = (M @ measurement).ravel()
        K = P_new @ C.T @ (np.linalg.inv(C @ P_new @ C.T + Q))

        x_est = x_pred + K @ (y - C @ x_pred)

        a = K @ C
        P_est = (np.eye(np.shape(a)[0]) - K @ C) @ P_new

        self.x = x_est
        self.P = P_est
        return x_est
