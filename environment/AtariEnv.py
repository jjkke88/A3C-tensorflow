import cv2
import numpy as np
from parameters import *


class AtariEnv(object):
    def __init__(self, env, screen_size=(84, 84)):
        self.env = env
        # constants
        self.screen_size = screen_size
        self.frame_skip = flags.frame_skip
        self.frame_seq = flags.frame_seq
        # local variables
        self.state = np.zeros(self.state_shape, dtype=np.float32)

    @property
    def state_shape(self):
        return [self.screen_size[0], self.screen_size[1], self.frame_seq]

    @property
    def action_dim(self):
        return self.env.action_space.n

    def precess_image(self, image):
        image = cv2.cvtColor(cv2.resize(image, self.screen_size), cv2.COLOR_BGR2GRAY)
        image = np.divide(image, 256.0)
        return image

    def reset_env(self):
        obs = self.env.reset()
        self.state[:, :, :-1] = 0
        self.state[:, :, -1] = self.precess_image(obs)
        return self.state

    def forward_action(self, action):
        obs, reward, done = None, None, None
        for _ in xrange(self.frame_skip):
            obs, reward, done, _ = self.env.step(action)
            if done:
                break
        obs = self.precess_image(obs)
        obs = np.reshape(obs, newshape=list(self.screen_size) + [1]) / 256.0
        self.state = np.append(self.state[:, :, 1:], obs, axis=2)
        # clip reward in range(-1, 1)
        reward = np.clip(reward, -1, 1)
        return self.state, reward, done