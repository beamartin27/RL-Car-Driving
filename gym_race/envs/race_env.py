import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_race.envs.pyrace_2d import PyRace2D

class RaceEnv(gym.Env):
    metadata = {'render_modes' : ['human'], 'render_fps' : 30}
    def __init__(self, render_mode="human", ):
        print("init")
        self.action_space = spaces.Discrete(3) # 3 possible actions: 0, 1, 2
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), dtype=int) # 5 sensors, each integer 0-10
        self.is_view = True
        self.pyrace = PyRace2D(self.is_view) # creates the actual game
        self.memory = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        mode = self.pyrace.mode
        del self.pyrace # destroy old game
        self.is_view = True
        self.msgs=[]
        self.pyrace = PyRace2D(self.is_view, mode = self.render_mode) # create fresh game (car back at start)
        obs = self.pyrace.observe() # get initial radar readings
        return np.array(obs),{}

    def step(self, action):
        self.pyrace.action(action)      # apply the action to the car
        reward = self.pyrace.evaluate() # compute reward
        done   = self.pyrace.is_done()  # check if episode is over
        obs    = self.pyrace.observe()  # get new radar readings
        return np.array(obs), reward, done, False, {'dist':self.pyrace.car.distance, 'check':self.pyrace.car.current_check, 'crash': not self.pyrace.car.is_alive}

    # def render(self, close=False , msgs=[], **kwargs): # gymnasium.render() does not accept other keyword arguments
    def render(self): # gymnasium.render() does not accept other keyword arguments
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def save_memory(self, file):
        # print(self.memory) # heterogeneus types
        # np.save(file, self.memory)
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # Currently used to save the full training history to disk.
