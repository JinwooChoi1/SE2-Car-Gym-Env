import numpy as np
from tools.BasePF import *

class ArtificialPF(BasePF):
    def __init__(self,obstacles,pot_weight,world_size = np.array([0,50,0,50]),target_pos = np.array([5,5,np.pi/4]), Qd_coeff = 5):
        """
        Initialize the ArtificialPF object.
        The parent of the ArtificialPF object is the BasePF object.
        Parameters:
        - target_pos (np.array): The position of the target.
        - obstacles (list): A list of obstacles dictionary or obstacles class.
        - pot_weight (np.array or list): The weight of the attractive and repulsive potential field.
        """
        super().__init__(world_size=world_size,target_pos = target_pos,obstacles = obstacles)
        self.pot_weight = pot_weight
        self.Qd_coeff = Qd_coeff

        
        x =  np.linspace(self.world_size[0],self.world_size[1],300)
        y = np.linspace(self.world_size[0],self.world_size[1],300)
        xx, yy = np.meshgrid(x,y,indexing='xy')
        self.max_attr_val = np.max((xx-target_pos[0])**2 + (yy-target_pos[1])**2)
        
    def attractor_field(self,x):

        # The attractor potential function is defined as the distance between the agent and the target.
        # If the dimension of the location is 2, transpose the location.
        # Try to vectorize the location and the value of the potential field.
        if x.ndim == 2:
            agent_location = x[:2,:]            
            target_location = np.array([self.target_pos[:2]]).transpose()
        else:
            agent_location = x[:2]
            target_location = np.array([self.target_pos[:2]]).squeeze()
            
        diff_location = agent_location-target_location

        pot = np.sum(diff_location**2,axis = 0)

        return pot
    
    def repulsive_field(self,x):
        # potential function : 1/2*(1/distance-1/Q)**2 if distance < Q

        if x.ndim == 2:
            agent_location = x[:2,:]
            pot = np.zeros(agent_location.shape[1])

        else:
            agent_location = x[:2]
            pot = 0.0

        for obstacle in self.obstacles:
            d = obstacle.get_min_distance(agent_location)
            Qd = np.mean(obstacle.size)*self.Qd_coeff
            if agent_location.ndim == 2:
                nonzero_idx = np.logical_not(np.isclose(d,0.0,atol= 1e-2))
                pot[nonzero_idx] += (d[nonzero_idx] <= Qd)*(1/d[nonzero_idx]-1/Qd)
            else:
                if not np.isclose(d,0.0,atol= 1e-2):
                    pot += (d <= Qd)*(1/d-1/Qd)

        return pot

    def zfun(self,agent_location = None):        
        attr_val = self.attractor_field(agent_location)
        rep_val = self.repulsive_field(agent_location)
        pot_val = self.pot_weight[0]*attr_val + self.pot_weight[1]*rep_val

        return pot_val