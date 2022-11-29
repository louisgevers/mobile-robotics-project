from src import model


def update_robot(
    robot: model.Robot, command: model.MotorSpeed, sensors: model.SensorReading
):
    ### Access the sent command (i.e. sent velocities to the motors)
    # command.left
    # command.right

    ### Access the read data
    # sensors.ground.left or sensors.ground.v[0]
    # sensors.ground.right or sensors.ground.v[1]

    # sensors.horizontal.left or sensors.horizontal.v[0]
    # sensors.horizontal.center_left or sensors.horizontal.v[1]
    # sensors.horizontal.center or sensors.horizontal.v[2]
    # sensors.horizontal.center_right or sensors.horizontal.v[3]
    # sensors.horizontal.right or sensors.horizontal.v[4]

    # sensors.motor.left
    # sensors.motor.right

    # ---

    ### Update robot pose (no need to return)
    # robot.angle =
    # robot.position.x =
    # robot.position.y =

    pass
    
class filter:
    def __init__(self,x0,P0,T):
        self.A=np.array([[1,0,0,0,0],
                         [0,1,0,0,0],
                         [0,0,1,0,0],
                         [0,0,0,0,0],
                         [0,0,0,0,0]])
        self.speed=[]
        self.x=x0
        self.u=[]
        self.P=P0
        self.T=T
        self.pictures=[]
        self.L=5
        self.R=None
        self.Q=None
    
    
    def Jacobian(self,x,Bu):
        return [[1,0,-Bu[1],0,0],
                [0,1,Bu[0],0,0],
                [0,0,1,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0]]
        
        
    def parameter(self,coord_ini,coord_end, orientation_ini, orientation_end):
        a=1
        #parameters required:
        #speed multiplier
        #angular speed multiplier
        #
        #Measurement y
        #H
        
        
        
        
        self.Q=Q
        self.R=R
        
        

    def Kalmanfilter(self,time,u,speed):
        #x=[x,y,theta,xdot,thetadot]
        #u=[vl,vr]
        x=self.x[-1]
        Sigma=self.P[-1]
        theta=x[2]
        B=0.5*np.dot(np.diag([np.cos(theta)*self.T,
                              np.sin(theta)*self.T,
                              1/self.L,
                              1,
                              1/self.L]),
                             [[1,1],
                              [1,1],
                              [-1,1],
                              [1,1],
                              [-1,1]])
        
        x_pred=self.A@x+B@u
        H=[[1,1],[-1,1]]
        G=self.Jacobian(x, B@u)
        P_new=G@Sigma@G.T+self.R
        
        K=P_new@H.T@(np.linalg.inv(H@P_new@H.T+self.Q))
                
        y=H@speed
        C=np.concatenate((np.zeros((2,3)),np.eye(2)),axis=1)
        
        x_est=x_pred+K@(y-C@x)
        Y=K@H
        P_est=(np.eye(np.shape(Y))-Y)@
        
        
        return x_est,P_est

    
