from dataclasses import dataclass, field

from src.trainer.agent import Agent
from src.trainer.model import ModelFactory

import tensorflow as tf
import keras
import numpy as np

import os

class ACAgent(Agent):
    @dataclass
    class Params(Agent.Params):
        model: ModelFactory.default_param_type() = field(default_factory = ModelFactory.default_params)

    def __init__(self, params, obs_space, act_space):
        super().__init__(params)
        print('params: ', params)
        print('obs_space: ', obs_space)
        print('act_space: ', act_space)
        self.actor = ModelFactory.create(params.model, obs_space=obs_space, act_space=act_space)
        self.critic = ModelFactory.create(params.model, obs_space=obs_space, act_space=None)

        self.expert_inference = False

    def save_network(self, path, name="model"):
        print('Agent path: ', f"{path}/actor_{name}.keras")
        print('Critic path: ', f"{path}/critic_{name}")
        self.actor.model.save(f"{path}/actor_{name}.keras")
        self.critic.model.save(f"{path}/critic_{name}.keras")

    def load_network(self, path, name="model"):
        print('load_network path: ', path)
        self.actor.model = keras.models.load_model(f"{path}/actor_{name}.keras")
        self.critic.model = keras.models.load_model(f"{path}/critic_{name}.keras")

    def load_weights(self, path, name="latest"):
        print('load_network path: ', path)
        self.actor.model.load_weights(f"{path}/actor_{name}.weights.h5")
        self.critic.model.load_weights(f"{path}/critic_{name}.weights.h5")

    def save_weights(self, path, name="latest"):
        self.actor.model.save_weights(f"{path}/actor_{name}.weights.h5")
        self.critic.model.save_weights(f"{path}/critic_{name}.weights.h5")

    def save_keras(self, path):
        self.actor.model.save(f"{path}/actor.keras")
        self.critic.model.save(f"{path}/critic.keras")

    def load_keras(self, path):
        print('Current Working Directory: ', os.getcwd())
        print('Finding for file in path: ', f"{path}/actor.keras")
        print('Is there a file in this path? ', os.path.isfile(f"{path}/actor.keras"))
        self.actor.model = tf.keras.models.load_model(f"{path}/actor.keras")
        self.critic.model = tf.keras.models.load_model(f"{path}/critic.keras")

    # @tf.function
    def actor_inference(self, obs):
        # print("mask in actor_inference:", obs.get("mask"))
        # print("obs keys:", obs.keys())
        if self.expert_inference:
            return self.actor.predict_expert(obs)
        else:
            mask = obs["mask"]
            logits = self.actor(obs)
            masked = tf.where(mask, logits, tf.fill(tf.shape(logits), -np.inf))
            return tf.nn.softmax(masked, axis=-1)

    @tf.function
    def critic_inference(self, obs):
        return self.critic.predict_expert(obs)[..., None] if self.expert_inference else self.critic(obs)

    @tf.function
    def get_probs_and_value(self, obs):
        probs = self.actor_inference(obs)
        value = self.critic_inference(obs)

        return probs, value

    # @tf.function(reduce_retracing=True)
    def get_action_prob_and_value(self, obs):
        # print('This is in ppo/agent.py get_action_prob_and_value!')
        # print("mask at entry:", obs.get("mask"))
        action, probs = self.get_exploration_action(obs)
        value = self.get_value(obs)
        return action, probs, value

    @tf.function
    def get_value(self, obs):
        value = self.critic_inference(obs)
        return tf.squeeze(value, axis=-1)

    # @tf.function
    def get_exploration_action(self, obs, step=None):
        # print('This is in ppo/agent.py get_exploration_action')
        probs = self.actor_inference(obs)
        actions = tf.random.categorical(tf.math.log(probs), 1)
        p = tf.gather_nd(probs, actions, batch_dims=1)
        actions = tf.squeeze(actions, axis=-1)
        return actions, p

    # @tf.function
    def get_exploitation_action(self, obs):
        probs = self.actor_inference(obs)
        action = tf.argmax(probs, axis=-1)
        return action, tf.ones_like(action)

    def summary(self):
        print("Actor:")
        self.actor.model.summary()
        print("Critic:")
        self.critic.model.summary()
