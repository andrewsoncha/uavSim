from dataclasses import dataclass, fields
from typing import Tuple

import numpy as np
from gymnasium import spaces

from utils import Factory

import skimage.measure
import random

import cv2

import traceback

class ObservationFunction:
    @dataclass
    class Params:
        position_history: bool = True
        random_layer: bool = False

    def __call__(self, state):
        return self.observe(state)

    def observe(self, state):
        raise NotImplementedError()

    def observe_multi(self, states):
        observes = [self.observe(state) for state in states]
        obs = {key: np.concatenate([observe[key] for observe in observes], axis=0) for key in observes[0].keys()}
        return obs

    def get_observation_space(self, state):
        raise NotImplementedError()


class PlainMapObservation(ObservationFunction):
    @dataclass
    class Params(ObservationFunction.Params):
        padding_values: Tuple[int] = (0, 1, 1, 0, 0, 0)
        pad_frame: bool = False

    def __init__(self, params: Params, max_budget):
        self.params = params
        self.max_budget = max_budget
        self.padded_map = None

    def observe(self, state):
        # print('Plain observation function called!')
        map_layers = state.map
        position_layer = np.zeros_like(state.map[..., 0])
        position_layer[state.position[0], state.position[1]] = 1
        map_layers = np.concatenate((map_layers, np.expand_dims(position_layer, -1)), axis=-1)
        if self.params.position_history:
            map_layers = np.concatenate((map_layers, np.expand_dims(state.position_history, -1)), axis=-1)
        if self.params.pad_frame:
            if self.padded_map is None:
                m = map_layers.shape[0] + 2
                self.padded_map = np.repeat(
                    np.repeat(np.reshape(self.params.padding_values, (1, 1, -1)), repeats=m, axis=0), repeats=m,
                    axis=1).astype(float)
            pm = self.padded_map.copy()
            pm[1:-1, 1:-1] = map_layers
            map_layers = pm

        map_layers = np.expand_dims(map_layers, axis=0)
        scalars = np.expand_dims(
            np.stack((state.budget / self.max_budget, state.landed), axis=-1), axis=0)
        mask = np.expand_dims(state.action_mask, axis=0)

        obs = {"map": map_layers, "scalars": scalars, "mask": mask}
        return obs

    def get_observation_space(self, state):
        obs = self.observe(state)
        return spaces.Dict(
            {
                "map": spaces.Box(low=0, high=1, shape=obs["map"].shape, dtype=float),
                "scalars": spaces.Box(low=0, high=1, shape=obs["scalars"].shape, dtype=float),
                "mask": spaces.Box(low=0, high=1, shape=obs["mask"].shape, dtype=bool)
            }
        )


class CenteredMapObservation(ObservationFunction):
    @dataclass
    class Params(ObservationFunction.Params):
        padding_values: Tuple[int] = (0, 1, 1, 0, 0)

    def __init__(self, params: Params, max_budget):
        self.params = params
        self.max_budget = max_budget

        self.centered_map = None
        self.centered_cover = None

    def pad_centered(self, map_layers, position):

        x, y = np.array(map_layers.shape[:2]) - position - 1
        m = map_layers.shape[0]
        # print('m: ', m)

        if self.centered_map is None:
            m_c = m * 2 - 1
            print('m_c: ', m_c)
            # print('np.reshape(self.params.padding_values, (1, 1, -1)), repeats=m_c, axis=0): ', np.reshape(self.params.padding_values, (1, 1, -1)).shape)

            self.centered_map = np.repeat(
                np.repeat(np.reshape(self.params.padding_values, (1, 1, -1)), repeats=m_c, axis=0), repeats=m_c,
                axis=1).astype(float)

        centered_map = self.centered_map.copy()
        # print('centered_map: ', centered_map.shape)
        # print('map_layers: ', map_layers.shape)
        centered_map[x:x + m, y:y + m] = map_layers
        # print('pad_centered  centered_map shape: ', centered_map.shape)

        return centered_map

    def pad_centered_patch(self, cover_layer, position):
        x, y = np.array(cover_layer.shape[:2]) - position - 1
        m = cover_layer.shape[0]
        # print('m: ', m)

        if self.centered_cover is None:
            m_c = m * 2 - 1
            self.centered_cover = np.zeros((m_c, m_c), dtype=bool)

        centered_cover = self.centered_cover.copy()
        centered_cover[x:x + m, y:y + m] = cover_layer

        return centered_cover

    def observe(self, state):
        map_layers = state.map
        # print('map_layers.shape: ', map_layers.shape)
        if self.params.position_history:
            map_layers = np.concatenate((map_layers, np.expand_dims(state.position_history, -1)), axis=-1)

        centered_map = np.expand_dims(self.pad_centered(map_layers, state.position), axis=0)
        centered_patch_cover = np.expand_dims(self.pad_centered_patch(state.patch_cover, state.position), axis=0)
                                            
        scalars = np.expand_dims(
            np.stack((state.budget / self.max_budget, state.landed), axis=-1), axis=0)
        mask = np.expand_dims(state.action_mask, axis=0)

        return {"map": centered_map, "scalars": scalars, "mask": mask, 'patch_cover': centered_patch_cover}

    def observe_multi(self, states):
        map_layers = [state.map for state in states]
        patch_covers = [state.patch_cover for state in states]

        if self.params.position_history:
            position_histories = [state.position_history for state in states]
            map_layers = [np.concatenate((maps, np.expand_dims(position_history, -1)), axis=-1) for
                          maps, position_history in zip(map_layers, position_histories)]

        centered_maps = np.stack([self.pad_centered(maps, state.position) for maps, state in zip(map_layers, states)], 0)
        scalars = np.stack([np.stack((state.budget / self.max_budget, state.landed), axis=-1) for state in states],
                           axis=0)

        centered_patch_covers = np.stack([self.pad_centered_patch(state.patch_cover, state.position) for patch_cover, state in zip(patch_covers, states)], 0)

        mask = np.stack([state.action_mask for state in states], axis=0)
        return {"map": centered_maps, "scalars": scalars, "mask": mask, 'patch_cover': centered_patch_covers}

    def get_observation_space(self, state):
        obs = self.observe(state)
        return spaces.Dict(
            {
                "map": spaces.Box(low=0, high=1, shape=obs["map"].shape, dtype=float),
                "scalars": spaces.Box(low=0, high=1, shape=obs["scalars"].shape, dtype=float),
                "mask": spaces.Box(low=0, high=1, shape=obs["mask"].shape, dtype=bool)
            }
        )

# Written by Andrew Chang
# Every cell in the map has a 'flip_prob' probability of flipping
# (0 -> 1, 1 -> 0) 
# This method only works when the map is binary (0 or 1)
# If it's a range of real number this might need to change
# Andrew Chang - June 29th, 2026
def flipMutate(global_map_arr, flip_prob: float):
    map_arr = global_map_arr.copy()
    map_shape = map_arr.shape
    assert len(map_shape) == 4
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            for k in range(map_shape[2]):
                c = map_arr[i, j, k, 3] # the 3rd value of the cell array is whether or
                                        # not the cell is a target zone
                # Flip c with the probability of flip_prob
                if random.random() < flip_prob:
                    c = 1.0 - c
                map_arr[i, j, k, 3] = c
    return map_arr 

# Written by Andrew Chang
# Consistent patches of 1 or 0 in both target and other dimensions
# patches are consistent in the global coordinates (if drone moves,
# the patches are seen moved in the opposite direction in the global
# view
# If it's a range of real number this might need to change
# Andrew Chang - July 25th, 2026.
def patchMutate(obs):
    map_arr = obs['map'].copy()
    map_shape = map_arr.shape
    map_target_int = np.array([map_arr[:,:,:, 3], map_arr[:,:,:, 3], map_arr[:,:,:, 3]], dtype=np.uint8)*255
    map_target_int = np.moveaxis(map_target_int, 0, -1)

    patch_cover = obs['patch_cover'].copy()
    patch_cover_int = np.array([patch_cover, patch_cover, patch_cover], dtype=np.uint8)*255
    patch_cover_int = np.moveaxis(patch_cover_int, 0, -1)
    assert len(map_shape) > 0 
    assert map_shape[:3] == patch_cover.shape[:3]
    for i in range(map_shape[0]): # Different Observation
        for j in range(map_shape[3]): # each map corresponding to different parts of map (obstacles, targets, NFZs, etc.)
            oneMap = map_arr[i, :, :, j]
            map_arr[i, :, :, j] = oneMap * np.invert(patch_cover[i])
    return map_arr

class GlobLocObservation(CenteredMapObservation):
    @dataclass
    class Params(CenteredMapObservation.Params):
        global_map_scaling: int = 3
        local_map_size: int = 17

    def __init__(self, params: Params, max_budget):
        super().__init__(params, max_budget)
        self.params = params

    def observe(self, state):
        obs = super().observe(state)
        obs = self._observe(obs)

        return obs

    def _observe(self, obs):
        centered = obs['map'].copy()
        mutated = patchMutate(obs)
        obs.pop('map')
        obs.pop('patch_cover')

        g = self.params.global_map_scaling
        l = self.params.local_map_size
        mutated_global_map = skimage.measure.block_reduce(mutated, (1, g, g, 1), np.mean)
        
        # Add Probability flip
        # mutated_global_map = flipMutate(global_map, 0.05)

        # diff = global_map - mutated_global_map
        # print("Changed Cells count: ", np.count_nonzero(diff))

        x, y = centered.shape[1:3]
        local_map = centered[:, x // 2 - l // 2: x // 2 + l // 2 + 1, x // 2 - l // 2: x // 2 + l // 2 + 1, :]
        obs.update({"global_map": mutated_global_map, "local_map": local_map})

        return obs

    def observe_multi(self, states):
        obs = super().observe_multi(states)
        obs = self._observe(obs)

        return obs

    def get_observation_space(self, state):
        obs = self.observe(state)
        return spaces.Dict(
            {
                "global_map": spaces.Box(low=0, high=1, shape=obs["global_map"].shape, dtype=np.float32),
                "local_map": spaces.Box(low=0, high=1, shape=obs["local_map"].shape, dtype=np.float32),
                "scalars": spaces.Box(low=0, high=1, shape=obs["scalars"].shape, dtype=np.float32),
                "mask": spaces.Box(low=0, high=1, shape=obs["mask"].shape, dtype=bool)
            }
        )


class ObservationFunctionFactory(Factory):
    @classmethod
    def registry(cls):
        return {
            "glob_loc": GlobLocObservation,
            "centered": CenteredMapObservation,
            "plain": PlainMapObservation
        }

    @classmethod
    def defaults(cls):
        return "glob_loc", GlobLocObservation
