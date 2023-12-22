import numpy as np
from tools.Obstacles import *

import matplotlib.pyplot as plt

# Base PF class
# The BasePF class is the base class for all PFs.
# It is used to define pot_field that are used in the Kinematic Env.
# __call__ method is used to calculate the PF at each point.
# In the kinematic env, use bpfname(agent_location) to calculate the PF at each point.
class BasePF():
    def __init__(self,world_size=None,target_pos=None,obstacles=None):
        self.name = "BasePF"

        self.world_size = np.array(world_size, dtype=np.float64)
        self.target_pos = np.array(target_pos, dtype=np.float64)
        self.obstacles = obstacles
    

        # add the boundary of the world to the obstacles.
        bounds_width = 0.5
        bounds_size = np.array([world_size[1]-world_size[0] + 2* bounds_width,
                                world_size[3]-world_size[2] + 2* bounds_width])
        bounds_center = np.array([(world_size[0]+world_size[1])/2,
                                  (world_size[2]+world_size[3])/2])
        obsts_list = [
            {'xyt': [bounds_center[0], world_size[2]-bounds_width/2, 0], 'size': [bounds_size[0], bounds_width]},
            {'xyt': [bounds_center[0], world_size[3]+bounds_width/2, 0], 'size': [bounds_size[0], bounds_width]},
            {'xyt': [world_size[0]-bounds_width/2, bounds_center[1], 0], 'size': [bounds_width,bounds_size[1]]},
            {'xyt': [world_size[1]+bounds_width/2, bounds_center[1], 0], 'size': [bounds_width,bounds_size[1]]}
        ]

        if isinstance(obstacles,list):
            obsts_list = obstacles + obsts_list
        elif isinstance(obstacles,Obstacles):
            for obst in obstacles.obsts_list:
                obsts_list.append(obst)
        
        # Setting up the obstacles (Define in Obstacle.py)
        self.obstacles = Obstacles(obsts_list)

    def __call__(self,x):
        return self.zfun(x)
    
    def zfun(self,x):
        pass

    def dzfun(self,x):
        pass

    def plot(self,ax):
        if ax is None:
            fig, ax = plt.subplots()
        
        x =  np.linspace(self.world_size[0],self.world_size[1],300)
        y = np.linspace(self.world_size[0],self.world_size[1],300)
        xx, yy = np.meshgrid(x,y,indexing='xy')

        zz = self.zfun(np.array([xx.flatten(),yy.flatten()]))
        zz = zz.reshape(xx.shape)
        ctr = ax.contourf(xx,yy,zz,cmap=plt.colormaps.get_cmap('coolwarm'))
        plt.colorbar(ctr,ax = ax)
        

        # If obstacles are defined, plot the obstacles.
        if self.obstacles != None:
            self.obstacles.plot(ax)

        return ctr