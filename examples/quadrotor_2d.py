import numpy as np
import gym
import gpflow
import pickle
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import QuadraticReward
import tensorflow as tf
from tensorflow import logging
from utils import rollout, policy, reward_wrapper
np.random.seed(0)

# Introduces a simple wrapper for the gym environment
# Reduces dimensions, avoids non-smooth parts of the state space that we can't model
# Uses a different number of timesteps for planning and testing
# Introduces priors


import gym
from gym import error, spaces, utils, logger
import numpy as np
from gym.utils import seeding

import os


class Quadrotor2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        # Quadrotor parameters
        self.m = 0.5   # mass
        self.Iy = 1e-2 # moment of inertia about y (body)
        self.kr = 1e-4 # rotational drag coefficient
        self.kt = 1e-4 # translational drag coefficient
        self.l = 0.125 # arm length

        self.dt = 0.1
        self.g = 9.81

        # Quadrotor state and reference signals
        self.x = None
        self.x_goal = np.array([0., 0., 0., 10., 0., 0.])
        self.x_dim = 6

        # Cost function parameters
        self.Q = np.diag([3e0, 1e0, 1e0, 1e0, 1e-1, 2e0])

        # Rendering
        self.viewer = None
        self.x_range = 10.

        # Actions are deviations from motor forces for hover (i.e. mg/2) (N)
        act_low_bounds = np.array([-0.96 * self.m * self.g,           # total thrust deviation (N)
                                   -1.23 * self.m * self.g * self.l]) # moment about pitch (Nm)
        act_high_bounds = np.array([1.5 * self.m * self.g,            # total thrust deviation (N)
                                    1.23 * self.m * self.g * self.l]) # moment about pitch (Nm)
        self.action_space = spaces.Box(low=act_low_bounds, high=act_high_bounds)
        # Observations are full 6D state, quadrotor bounded in 20m x 20m box
        obs_low_bounds = np.array([-np.pi, # pitch angle
                                   -3.,    # pitch rate
                                   -10.,   # x position (inertial)
                                   0.,     # z position (inertial)
                                   -5.,    # x velocity (inertial)
                                   -5.])   # z velocity (inertial)
        obs_high_bounds = np.array([np.pi - np.finfo(np.float32).eps, # pitch angle
                                    3.,                               # pitch rate
                                    10.,                              # x position (inertial)
                                    20.,							  # z position (inertial)
                                    5.,  							  # x velocity (inertial)
                                    5.])							  # z velocity (inertial)
        self.observation_space = spaces.Box(low=obs_low_bounds, high=obs_high_bounds)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_goal(self, goal):
        assert type(goal) == np.ndarray
        assert goal.shape == (self.x_dim,)
        self.x_goal = goal

    def step(self, action):
        # Correction to nominal input (hover)
        T = action[0] + self.m * self.g # total thrust (N)
        My = action[1]                  # moment about pitch (Nm)

        # 2D quadrotor dynamics model following Freddi, Lanzon, and Longhi, IFAC 2011
        x_cur = self.x
        x_dot = np.empty(x_cur.shape)
        x_dot[0] = x_cur[1]
        x_dot[1] = -self.kr/self.Iy*x_cur[1] + 1/self.Iy*My
        x_dot[2] = x_cur[4]
        x_dot[3] = x_cur[5]
        x_dot[4] = -self.kt/self.m*x_cur[4] + 1/self.m*np.sin(x_cur[0])*T
        x_dot[5] = -self.kt/self.m*x_cur[5] + 1/self.m*np.cos(x_cur[0])*T - self.g

        x_next = x_cur + x_dot * self.dt
        # Wrap angle
        if x_next[0] > np.pi:
            x_next[0] -= 2*np.pi
        elif x_next[0] < -np.pi:
            x_next[0] += 2*np.pi

        self.x = x_next
        e = self.x - self.x_goal
        # Correct angle error
        if e[0] > np.pi:
            e[0] -= 2*np.pi
        elif e[0] < -np.pi:
            e[0] += 2*np.pi

        done = False
        reward = -0.5 * (e.T @ self.Q @ e)

        return self.x, reward, done, {}

    def reset(self):
        print("Environment reset")
        # self.x = self.np_random.uniform(low=-0.5, high=0.5, size=self.observation_space.shape)
        self.x = np.zeros(6)
        # self.x[0] = self.np_random.uniform(low=-5*np.pi/180., high=5*np.pi/180.)   # spawn at some random small pitch angle
        self.x[2] = self.np_random.uniform(low=-1, high=1)                           # spawn at some x location
        self.x[3] = self.np_random.uniform(low=9, high=11)                           # spawn at some height in the middle
        return self.x

    def render(self, mode='human', close=False):
        screen_width = 800
        screen_height = 800

        world_width = self.x_range*2
        scale = screen_width/world_width
        ref_size = 5.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # Draw reference
            ref = rendering.make_circle(ref_size)
            self.reftrans = rendering.Transform()
            ref.add_attr(self.reftrans)
            ref.set_color(1,0,0)
            self.viewer.add_geom(ref)
            # Draw start
            start = rendering.make_circle(ref_size)
            self.starttrans = rendering.Transform()
            start.add_attr(self.starttrans)
            start.set_color(0,0,0)
            self.viewer.add_geom(start)
            # Draw drone
            dir_path = os.path.dirname(os.path.realpath(__file__))
            quad = rendering.Image('%s/quadrotor2d.png' % dir_path, 60, 12)
            self.quadtrans = rendering.Transform()
            quad.add_attr(self.quadtrans)
            self.viewer.add_geom(quad)

        if self.x is None:
            return None

        quad_x = self.x[2]*scale+screen_width/2.0 
        quad_y = self.x[3]*scale 
        self.quadtrans.set_translation(quad_x, quad_y)
        self.quadtrans.set_rotation(-self.x[0])

        y = self.x_goal[2:4]
        ref_x = y[0]*scale+screen_width/2.0
        ref_y = y[1]*scale
        self.reftrans.set_translation(ref_x, ref_y)

        y = np.array([0, 10])
        start_x = y[0]*scale+screen_width/2.0
        start_y = y[1]*scale
        self.starttrans.set_translation(start_x, start_y)

        return self.viewer.render(return_rgb_array=(mode =='rgb_array'))
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def save_pilco(path, X, Y, pilco, sparse=False):
    np.savetxt(os.path.join(path, 'X.csv'), X, delimiter=',')
    np.savetxt(os.path.join(path, 'Y.csv'), Y, delimiter=',')
    if sparse:
        with open(os.path.join(path, 'n_ind.txt'), 'w') as f:
            f.write('%d' % pilco.mgpr.num_induced_points)
            f.close()
    np.save(os.path.join(path, 'pilco_values.npy'), pilco.read_values())
    for i,m in enumerate(pilco.mgpr.models):
        np.save(os.path.join(path, "model_" + str(i) + ".npy"), m.read_values())

def load_pilco(path, sparse=False):
    X = np.loadtxt(os.path.join(path, 'X.csv'), delimiter=',')
    Y = np.loadtxt(os.path.join(path, 'Y.csv'), delimiter=',')
    if not sparse:
        pilco = PILCO(X, Y)
    else:
        with open(os.path.join(path, 'n_ind.txt'), 'r') as f:
            n_ind = int(f.readline())
            f.close()
        pilco = PILCO(X, Y, num_induced_points=n_ind)
    params = np.load(os.path.join(path, 'pilco_values.npy')).item()
    pilco.assign(params)
    for i,m in enumerate(pilco.mgpr.models):
        values = np.load(os.path.join(path, "model_" + str(i) + ".npy")).item()
        m.assign(values)
    return pilco


SUBS = 1
bf = 40
maxiter=80
state_dim = 6
control_dim = 2
max_action=1.0 # actions for these environments are discrete
target = np.array([0., 0., 5., 10., 0., 0.])
weights = np.diag([3e0, 1e0, 1e0, 1e0, 1e-1, 2e0])
m_init = np.zeros(state_dim)[None, :]
S_init = 0.01 * np.eye(state_dim)
T = 20
J = 1
N = 50
T_sim = 51
restarts = 0

with tf.Session() as sess:
    env = Quadrotor2DEnv()
    env.set_goal(target)

    # Initial random rollouts to generate a dataset
    X,Y = rollout(env, None, timesteps=T, random=True, SUBS=SUBS)
    for i in range(1,J):
        X_, Y_ = rollout(env, None, timesteps=T, random=True, SUBS=SUBS, verbose=True)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))

    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim

    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)

    R = QuadraticReward(state_dim=state_dim, x_goal=target, Q=weights)

    pilco = PILCO(X, Y, controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

    obs = None
    # for numerical stability
    for model in pilco.mgpr.models:
        # model.kern.lengthscales.prior = gpflow.priors.Gamma(1,10) priors have to be included before
        # model.kern.variance.prior = gpflow.priors.Gamma(1.5,2)    before the model gets compiled
        model.likelihood.variance = 0.001
        model.likelihood.variance.trainable = False

    for rollouts in range(N):
        print("**** ITERATION no", rollouts, " ****")
        pilco.optimize_models(maxiter=maxiter, restarts=restarts)
        pilco.optimize_policy(maxiter=maxiter, restarts=restarts)

        X_new, Y_new = rollout(env, pilco, timesteps=T_sim, verbose=True, SUBS=SUBS)

        # Since we had decide on the various parameters of the reward function
        # we might want to verify that it behaves as expected by inspection
        cur_rew = 0
        for t in range(0,len(X_new)):
            cur_rew += reward_wrapper(R, X_new[t, 0:state_dim, None].transpose(), 0.0001 * np.eye(state_dim))[0]
        print('Episode reward: {}'.format(cur_rew))

        # Update dataset
        X = np.vstack((X, X_new[:T, :])); Y = np.vstack((Y, Y_new[:T, :]))
        pilco.mgpr.set_XY(X, Y)

        if obs is None:
            obs = np.expand_dims(X_new, axis=0)
        else:
            obs = np.vstack([obs, np.expand_dims(X_new, axis=0)])

    file_path = os.path.join(os.getcwd(), "rollout_data.npz")
    np.savez(file_path, data=obs)

    model_path = os.path.join(os.getcwd(), "models")
    save_pilco(model_path, X, Y, pilco)