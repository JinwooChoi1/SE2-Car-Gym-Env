import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

from environments.DiffDriveEnv import *
from environments.BicycleEnv import *
from tools.Obstacles import *
from tools.ArtificialPF import *

# example code:
if __name__ == "__main__":

    # the dictionary to contain the obstacle's information.
    obsts_list = [
        {'xyt': [25, 25, 0], 'size': [5, 5]},
        {'xyt': [20, 10, 0.5], 'size': [4.5, 4]},
        {'xyt': [9.5, 30, 0], 'size': [2, 30]}
    ]
    
    # The artificial potential field needs :
    # world_size, [x_min,x_max,y_min,y_max]
    # target_pos, [x_target,y_target,theta_target]
    # obstacles, Obstacles object or the dictionary list.
    # pot_wieght, the weights of attractive field and repulsive field.
    apf = ArtificialPF(world_size=[0,50,0,50],target_pos=[45,45,0], obstacles=obsts_list, pot_weight=[1.0,20.0])

    # The differential drive car needs :
    # agent's init_pos, [x_initial,y_initial,theta_initial]    
    # the map of the world pot_field, any PF objects.
    # the reward weights
    # render_mode 'rgb_array' returns rgb array.
    dd_env = DiffDriveEnv(init_pos = [5,5,0], pot_field=apf, reward_weight=[1.0, 25.0, 1.0, 100.0, 100.0], render_mode='rgb_array')
    dd_env = gym.wrappers.RecordVideo(dd_env, 'videos')
    dd_env.reset()

    # plot with the potential field.
    dd_env.plot(True)

    observation, reward, terminated, _, _ = dd_env.step([1, 0])

    for i in range(100):
        observation, reward, terminated, _, _ = dd_env.step([0.5, 0.5])   
        print(f"state:{observation}, reward:{reward}, isdone:{terminated}")

    dd_env.close()

    # render_mode 'human' uses pygame.
    bi_env = BicycleEnv(init_pos = [5,5,0], pot_field=apf, reward_weight=[1.0, 25.0, 1.0, 100.0, 100.0], render_mode='human')
    bi_env.reset()

    observation, reward, terminated, _, _ = bi_env.step([1, 0])

    for i in range(100):
        observation, reward, terminated, _, _ = bi_env.step([0.25, 0.02])
        print(f"state:{observation}, reward:{reward}, isdone:{terminated}")

    bi_env.close()