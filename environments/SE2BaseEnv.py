import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pygame

# Importing the Gym library for creating the environment
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from gymnasium.envs.registration import register

from tools.Obstacles import *
from tools.SE2 import *

register(
     id="SE2_Env",
     entry_point="gym_examples.envs:SE2Env",
     max_episode_steps=1000,
)

# Defining a SE2 environment class based on the Gym environment
class SE2BaseEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, pot_field, init_pos, agent_size, reward_weight, suc_tol, render_mode):
        """
        Initialize the KinematicEnv environment.

        Parameters:
        - pot_field (BasePF): Potential Field class.
        - agent_size (list or np.array): Size of the agent.
        - init_pos (list or np.array): Initial position of the agent.
        - reward_weight (list or np.array): Weighting factors for rewards.
            1. the action cost.
            2. the potential field cost.
            3. the agent stationary cost. (based on the average point)
            4. the obstacle collision cost.
            5. the goal reward.
        - suc_tol (float): The tolerance for the success (reach the target).
        - render_mode (str): The rendering mode for the gym environment.
        """
        self.pot_field = pot_field
        self.world_size = self.pot_field.world_size
        self.obstacles = self.pot_field.obstacles
        
        self.agent_size = agent_size

        # Setting up the observation space SE(2).
        lowbound = np.array([self.world_size[0], self.world_size[2], 0])
        highbound = np.array([self.world_size[1], self.world_size[3], 2 * np.pi])
        self.observation_space = Dict(
            {
                "agent": Box(shape=(3,), low=lowbound, high=highbound),
                "target": Box(shape=(3,), low=lowbound, high=highbound)
            }
        )
        self._init_pos = np.array(init_pos)
        self._target_pos = np.array(self.pot_field.target_pos)
        
        # Defining reward weights
        # First element: for the gait cost
        # Second element: for the distance to target
        self.reward_weight = np.array(reward_weight)

        # Flag to track whether the environment has been reset
        self.isreset = False
        
        # The tolerance for the success (reach the target).
        self.suc_tol = suc_tol

         # The size of the PyGame window
        self.window_size = 512 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None


    def _get_obs(self):
        """
        Get the current observation.

        Returns:
        - dict: Dictionary containing "agent" and "target" observations.
        """
        agent_location = self._agent_SE2.vec_SE2()
        target_location = self._target_SE2.vec_SE2()
        return {"agent": agent_location, "target": target_location}

    def _get_info(self):
        """
        Get additional information about the current state.

        Returns:
        - dict: Dictionary containing information about the distance between agent and target.
        """
        # Calculating distance between agent and target
        location = self._get_obs()
        distance = np.sqrt(np.sum((location["agent"][:2] - location["target"][:2]) ** 2))

        return {"distance": distance}

    def reset(self, seed=None, options=None):
        """
        Reset the environment to a new episode.

        Parameters:
        - seed: Seed for the random number generator.
        - options: Additional options.

        Returns:
        - tuple: Initial observation and information.
        """
        self.isreset = True

        # Seeding the random number generator
        super().reset(seed=seed)

        # Choosing random locations for the agent and target
        self._agent_SE2 = SE2(self._init_pos)
        self._target_SE2 = SE2(self._target_pos)

        # Initializing history arrays
        self.pos_history = np.array([self._agent_SE2.vec_SE2()])
        self.action_history = np.array([])

        # Getting initial observation and information
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode in self.metadata["render_modes"]:
            self._render_frame()

        return observation, info

    
    def step(self, action):
        """
        Take a step in the environment given an action.

        Parameters:
        - action: Action to take.

        Returns:
        - tuple: New observation, reward, termination flag, additional info.
        """
        assert self.isreset, f"Environment should be reset, {self.isreset}"

        # Checking if the action is within bounds
        action_SE2, action_cost = self.get_action_SE2(action)

        info = self._get_info()

        # Calculating potential field value at the current state.
        agent_pos =  np.array(self._agent_SE2.vec_SE2())
        pot_value = self.pot_field(agent_pos)

        # Updating agent's state based on the selected motion primitive
        self._agent_SE2 = self._agent_SE2 @ action_SE2
        
        # Calculating potential field value at the next state.
        agent_pos =  np.array(self._agent_SE2.vec_SE2())
        next_pot_value = self.pot_field(agent_pos)      

        info = self._get_info()
        observation = self._get_obs()

        # Checking if the episode is terminated (agent reaches the target)
        terminated = np.isclose(info["distance"], 0, atol=self.suc_tol)

        # Reward calculation
        if not terminated:
            reward = 0            
            agent_obj = Obstacle(self._agent_SE2.vec_SE2(),self.agent_size,'k')

            reward -= self.reward_weight[0] * action_cost
            reward += self.reward_weight[1] * (pot_value - next_pot_value)
            if self.pos_history.shape[0] > 50:
                reward -= self.reward_weight[2] * np.linalg.norm(agent_pos - np.mean(self.pos_history[-51:,:],axis=0))

            if self.obstacles.is_collision(agent_obj):
                reward -= self.reward_weight[3]
                terminated = True
        else:
            reward = self.reward_weight[4]
       
        # Updating history arrays
        self.pos_history = np.append(self.pos_history, [self._agent_SE2.vec_SE2()], axis=0)
        if self.action_history.size == 0:
            self.action_history = np.array([action])
        else:
            self.action_history = np.append(self.action_history, np.array([action]), axis=0)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info    
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        ratio_coord_to_window = np.array(
            [self.window_size/(self.world_size[1]-self.world_size[0]),
            self.window_size/(self.world_size[3]-self.world_size[2])])

        # render the agent.
        agent_size = self.agent_size*ratio_coord_to_window
        agent_location = self._flip_and_scale_object(self._agent_SE2.vec_SE2()[:2],ratio_coord_to_window)
        agent_angle = self._agent_SE2.vec_SE2()[2]    

        agent_surf = pygame.Surface(np.int32(agent_size))
        agent_surf.set_colorkey((255,255,255))
        agent_surf.fill((0,0,0))
        agent_surf = pygame.transform.rotate(agent_surf,agent_angle*180.0/np.pi)
        agent_rect = agent_surf.get_rect(center = tuple(agent_location))        
        
        # render the target.
        target_location = self._flip_and_scale_object(self._target_SE2.vec_SE2()[:-1],ratio_coord_to_window)
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            target_location,
            ratio_coord_to_window[0],
        )

        if self.pos_history.shape[0] > 1:
            points = []
            for i in range(self.pos_history.shape[0]):                
                points.append(self._flip_and_scale_object((self.pos_history[i,0],self.pos_history[i,1]),ratio_coord_to_window))
            pygame.draw.lines(canvas, (50,50,50), False, points, 2)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())            
            self.window.blit(agent_surf,agent_rect)
            self.obstacles.render(self.window, ratio_coord_to_window)
            pygame.event.pump()    
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            # self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":  # rgb_array
            canvas.blit(agent_surf, agent_rect)        
            self.obstacles.render(canvas, ratio_coord_to_window)
            rgb_array = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

            return rgb_array
        
    def _flip_and_scale_object(self,point,ratio):
        point = point * ratio
        point[1] = self.window_size - point[1]
        return point

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


    def plot(self,ispotential = False):
        """
        Plot the trajectory of the agent and target.
        Parameters:
        - ispotential (bool): If True, plot the potential contour.
        - isdpotential (bool): If True, plot the potential vector field.
        """
        
        fig, ax = plt.subplots()

        ax.axis('equal')
        ax.set_xlabel('x')
        ax.set_xlabel('y')

        if ispotential:
            self.pot_field.plot(ax)
        
        # Plotting the robot's trajectory.
        pos = np.reshape(self.pos_history[0, :], [3, 1])
        n = 5
        for i in range(1, self.pos_history.shape[0]):
            action = self.action_history[i - 1, :]
            action_SE2, _ = self.get_action_SE2(action)
            for j in range(n):
                next_pos = (SE2(self.pos_history[i - 1, :]) @ (action_SE2 * (float(j + 1) / float(n)))).vec_SE2(1)
                pos = np.append(pos, next_pos, axis=1)
        ax.plot(pos[0, :], pos[1, :], 'k--', label='system trajectory')

        target_location = self._target_SE2.vec_SE2()

        # Plotting the position of each state and target
        ax.plot(target_location[0], target_location[1], 'ro', label='target')

        ax.quiver(target_location[0], target_location[1], 
                np.cos(target_location[2]), np.sin(target_location[2]), \
                color='r')

        ax.legend()
        plt.show()

        return fig, ax    
    
    def clip_action(self,action):
        pass

    def get_action_SE2(self,action):
        pass