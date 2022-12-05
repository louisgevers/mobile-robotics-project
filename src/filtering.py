from src import model


import random
import numpy as np
import matplotlib.pyplot as plt
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
        B=0.5*B
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
            M[3:,3:]=0.5*np.array([[1,1],[-1,1]])
            C=np.eye(5)
        else:
            measurement=speed
            M=[[0.5,0.5],[-0.5,0.5]]
            C=np.concatenate((np.zeros((2,3)),np.eye(2)),axis=1)
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
R=np.eye(5)
# R[3:,3:]=np.zeros((2,2))
picture=True
speedvar=2
posvar=0.1
thetavar=0.01
measvar=0.1
statevar=0.2
iter=30
if picture:
    i=5
    
else:
    i=2
x=[]
y=filter([0,0,0,0,0],np.zeros((5,5)),measvar*np.eye(i),statevar*R,0.1)
speeds=np.array([12,12])
coord=np.array([0,0,0])
for i in range(iter):
    xnext,Pnext=y.Kalmanfilter(speeds,np.random.normal(speeds,[speedvar,speedvar]),coord,picture)
    coord=np.random.normal(xnext[:3],[posvar,posvar,thetavar])
        
    x.append(xnext.tolist())
    # print(xnext)
    # print(Pnext)
speeds=np.array([11,-11])
for i in range(iter):
    xnext,Pnext=y.Kalmanfilter(speeds,np.random.normal(speeds,[speedvar,speedvar]),coord,picture)
    coord=np.random.normal(xnext[:3],[posvar,posvar,thetavar])
    
    x.append(xnext.tolist())
    # print(Pnext)
speeds=np.array([18,9])
for i in range(iter):
    xnext,Pnext=y.Kalmanfilter(speeds,np.random.normal(speeds,[speedvar,speedvar]),coord,picture)
    coord=np.random.normal(xnext[:3],[posvar,posvar,thetavar])
    
    x.append(xnext.tolist())
    # print(Pnext[1,0])
x=np.array(x)
print(x)
plt.plot(x[:,0],x[:,1])
plt.show()

    
