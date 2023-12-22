import matplotlib.pyplot as plt

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from environments.SE2BaseEnv import *

# Importing the Gym library for creating the environment
from gymnasium.spaces import Box

class DiffDriveEnv(SE2BaseEnv):
    def __init__(self,
                pot_field, 
                init_pos,
                agent_size = np.array([3,1]),
                wheel_speed_limit = np.array([-1,1,-1,1]),
                wheel_size = 0.5,
                reward_weight=[1.0, 10.0, 1.0, 100.0, 100.0],
                suc_tol=0.5,
                render_mode=None):

        super().__init__(pot_field, init_pos, agent_size, reward_weight, suc_tol, render_mode)
        self.wheel_size = wheel_size

        # Setting up the action space
        self.action_space = Box(
            low = np.array([wheel_speed_limit[0], wheel_speed_limit[2]]),
            high = np.array([wheel_speed_limit[1], wheel_speed_limit[3]]),
            dtype=np.float32)
    
    # clipping the action to the action space.
    def clip_action(self,action):        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def get_action_SE2(self,action):

        action = self.clip_action(action)

        # forward velocity = wheel_size * (left wheel velocity + right wheel velocity) / 2
        v = self.wheel_size*(action[0] + action[1])/2

        # angular velocity = wheel_size * (left wheel velocity - right wheel velocity) / robot's width
        omega = self.wheel_size*(action[0] - action[1])/self.agent_size[1]

        # Transforming action indices to motion primitive using the predefined mapping
        action_xyt = np.array([v, 0, omega])
        action_xyt = SE2.se2_to_SE2(action_xyt)
        action_SE2 = SE2(action_xyt)

        return action_SE2, 1