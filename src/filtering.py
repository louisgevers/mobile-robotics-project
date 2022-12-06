from src import model


import random
import numpy as np
import matplotlib.pyplot as plt

def update_robot(
    robot: model.Robot, command: model.MotorSpeed, sensors: model.SensorReading
):
    pass
class filter:
    def __init__(self,x0,P0,Q,R,T):
        
        self.x=[x0]
        self.u=[]
        self.P=[P0]
        self.T=T
        self.speed=[]
        self.camerapos=[]
        self.L=1
        self.R=R
        self.Q=Q
        self.u_prev=np.array([0,0])
        self.speedconv=0.434782608
        
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
        
        

    def Kalmanfilter(self,u,speed,camerapos,camera=False):
        #x=[x,y,theta,xdot,thetadot]
        #u=[vl,vr]
        
        udiff=u-self.u_prev
        self.u_prev=u
        # print(udiff)
        x=np.array(self.x[-1]).T
        Sigma=np.array(self.P[-1]).T
        theta=x[2]
        A=np.array([[1,0,0,np.cos(theta)*self.T,0],
                    [0,1,0,np.sin(theta)*self.T,0],
                    [0,0,1,0,self.T/self.L],
                    [0,0,0,1,0],
                    [0,0,0,0,1]])

        
        B=np.zeros((5,2))
        B[3:]=[[1,1],[-1,1]]
        B=self.speedconv*0.5*B
        # print(B)
        x_pred=A@x+B@udiff
        # print(A@x)
        # print(x_pred)
        G=np.array([[1,0,-0.5*self.T*np.sin(theta)*x[3],0.5*np.cos(theta)*self.T,0],
           [0,1,0.5*self.T*np.cos(theta)*x[4],0.5*np.sin(theta)*self.T,0],
           [0,0,1,0,self.T*0.5/self.L],
           [0,0,0,1,0],
           [0,0,0,0,1]])
        # print(G)
        # print(Sigma)
        P_new=G@Sigma@G.T +self.R
        P_est=P_new
        # print(P_new)
        if camera:
            # print(camerapos,speed)
            measurement=np.append(camerapos,speed)
            M=np.eye(5)
            M[3:,3:]=self.speedconv*0.5*np.array([[1,1],[-1,1]])
            C=np.eye(5)
        else:
            measurement=speed
            M=[[0.5,0.5],[-0.5,0.5]]
            C=self.speedconv*np.concatenate((np.zeros((2,3)),np.eye(2)),axis=1)
        y=M@measurement
        K=P_new@C.T@(np.linalg.inv(C@P_new@C.T+self.Q))
        # C=np.concatenate((np.zeros((2,3)),np.eye(2)),axis=1)

        x_est=x_pred+K@(y-C@x_pred)
        
        # print(y)
        # print(C)
        # print(C@x)
        a=K@C
        P_est=(np.eye(np.shape(a)[0])-K@C)@P_new

        self.x.append(x_est.tolist())
        self.P.append(P_est.tolist())
        return x_est,P_est

