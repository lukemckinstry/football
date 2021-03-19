# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Allows different types of players to play against each other."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import importlib
from absl import logging

from gfootball.env import config as cfg
from gfootball.env import constants
from gfootball.env import football_action_set
from gfootball.env import football_env_core
from gfootball.env import observation_rotation
import gym
import numpy as np
import itertools

import datetime, os, copy
import pickle
#from sklearn.linear_model import LogisticRegression
import tensorflow.compat.v1 as tf

import tensorflow.compat.v1.keras

from tensorflow.compat.v1.keras import Sequential
from tensorflow.compat.v1.keras.layers import Dense, InputLayer, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D



class FootballEnv(gym.Env):
  """Allows multiple players to play in the same environment."""

  def __init__(self, config):

    self._config = config
    player_config = {'index': 0}
    # There can be at most one agent at a time. We need to remember its
    # team and the index on the team to generate observations appropriately.
    self._agent = None
    self._agent_index = -1
    self._agent_left_position = -1
    self._agent_right_position = -1
    self._players = self._construct_players(config['players'], player_config)
    self._env = football_env_core.FootballEnvCore(self._config)
    self._num_actions = len(football_action_set.get_action_set(self._config))
    self._cached_observation = None
    self._supervised_log_filename = "episode" + str(datetime.datetime.now())
    #self._trained_supervised_lr_models = self._load_trained_supervised_lr_models()
    #self._trained_neural_network_model = self._load_trained_neural_network_model()
    self._trained_cnn_timeseries_model = self._load_trained_cnn_timeseries_model()

  @property
  def action_space(self):
    if self._config.number_of_players_agent_controls() > 1:
      return gym.spaces.MultiDiscrete(
          [self._num_actions] * self._config.number_of_players_agent_controls())
    return gym.spaces.Discrete(self._num_actions)

  def _load_trained_supervised_lr_models(self):
    models = []
    for idx in range(19):
      loaded_model = pickle.load(open('trained_models/logistic_regression_models/model' + str(idx) + '.sav', 'rb'))
      models.append(loaded_model)
    return models 

  def _load_trained_neural_network_model(self):
    model = Sequential()
    model.add(Dense(12, input_dim=26, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(19, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights('trained_models/nn_weights_only/model_weights.h5')
    return model

  def _load_trained_cnn_timeseries_model(self):
    model = Sequential([
      InputLayer((75,1)),
      Conv1D(filters=64, kernel_size=3, padding="same"),
      BatchNormalization(),
      ReLU(),
      Conv1D(filters=64, kernel_size=3, padding="same"),
      BatchNormalization(),
      ReLU(),
      Conv1D(filters=64, kernel_size=3, padding="same"),
      BatchNormalization(),  
      ReLU(),
      GlobalAveragePooling1D(),
      Dense(19, activation="softmax")
      ])
    model.load_weights('trained_models/cnn_timeseries_weights_only/cnn_model_weights.h5')    
    return model

  def _construct_players(self, definitions, config):
    result = []
    left_position = 0
    right_position = 0
    for definition in definitions:
      (name, d) = cfg.parse_player_definition(definition)
      config_name = 'player_{}'.format(name)
      if config_name in config:
        config[config_name] += 1
      else:
        config[config_name] = 0
      try:
        player_factory = importlib.import_module(
            'gfootball.env.players.{}'.format(name))
      except ImportError as e:
        logging.error('Failed loading player "%s"', name)
        logging.error(e)
        exit(1)
      player_config = copy.deepcopy(config)
      player_config.update(d)
      player = player_factory.Player(player_config, self._config)
      if name == 'agent':
        assert not self._agent, 'Only one \'agent\' player allowed'
        self._agent = player
        self._agent_index = len(result)
        self._agent_left_position = left_position
        self._agent_right_position = right_position
      result.append(player)
      left_position += player.num_controlled_left_players()
      right_position += player.num_controlled_right_players()
      config['index'] += 1
    return result

  def _convert_observations(self, original, player,
                            left_player_position, right_player_position):
    """Converts generic observations returned by the environment to
       the player specific observations.

    Args:
      original: original observations from the environment.
      player: player for which to generate observations.
      left_player_position: index into observation corresponding to the left
          player.
      right_player_position: index into observation corresponding to the right
          player.
    """
    observations = []
    for is_left in [True, False]:
      adopted = original if is_left or player.can_play_right(
      ) else observation_rotation.flip_observation(original, self._config)
      prefix = 'left' if is_left or not player.can_play_right() else 'right'
      position = left_player_position if is_left else right_player_position
      for x in range(player.num_controlled_left_players() if is_left
                     else player.num_controlled_right_players()):
        o = {}
        for v in constants.EXPOSED_OBSERVATIONS:
          o[v] = copy.deepcopy(adopted[v])
        assert (len(adopted[prefix + '_agent_controlled_player']) == len(
            adopted[prefix + '_agent_sticky_actions']))
        o['designated'] = adopted[prefix + '_team_designated_player']
        if position + x >= len(adopted[prefix + '_agent_controlled_player']):
          o['active'] = -1
          o['sticky_actions'] = []
        else:
          o['active'] = (
              adopted[prefix + '_agent_controlled_player'][position + x])
          o['sticky_actions'] = np.array(copy.deepcopy(
              adopted[prefix + '_agent_sticky_actions'][position + x]))
        # There is no frame for players on the right ATM.
        if is_left and 'frame' in original:
          o['frame'] = original['frame']
        observations.append(o)
    return observations

  def _action_to_list(self, a):
    if isinstance(a, np.ndarray):
      return a.tolist()
    if not isinstance(a, list):
      return [a]
    return a

  def _log_train_data(self,training_obs):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    logfile = open("supervised_logs/"+self._supervised_log_filename,"a")
    training_obs_string = ','.join([str(i) for i in training_obs])
    logfile.write(training_obs_string)
    logfile.write('\n')
    logfile.close()

  def _format_rnn_obs(self,obs,action):
    obs_copy = copy.deepcopy(obs)
    obs_copy.pop('frame')

    ## ball 
    ball_arr = [obs_copy['ball'],obs_copy['ball_direction'],obs_copy['ball_rotation']]
    ball = list(itertools.chain(*ball_arr))

    #designated player
    lt_des_player = obs_copy['left_team_designated_player']    
    lt_dp_posx, lt_dp_posy = obs_copy['left_team'][lt_des_player]
    lt_dp_velx, lt_dp_vely = obs_copy['left_team_direction'][lt_des_player]

    ## left team attacking 
    lt_pos = list(itertools.chain(*obs_copy['left_team'])) 
    lt_vel = list(itertools.chain(*obs_copy['left_team_direction']))
    lt_active = [int(i) for i in obs_copy['left_team_active']]
    lt_zip = list(zip(lt_pos,lt_vel,lt_active))
    lt_comp = list(itertools.chain(*lt_zip))

    #right team
    rt_pos = list(itertools.chain(*obs_copy['right_team']))
    rt_vel = list(itertools.chain(*obs_copy['right_team_direction']))
    rt_active = [int(i) for i in obs_copy['right_team_active']]
    rt_zip = list(zip(rt_pos,rt_vel,rt_active))
    rt_comp = list(itertools.chain(*rt_zip))

    pos_obs = [
      ball, lt_comp, rt_comp
    ]

    flat_obs = list(itertools.chain(*pos_obs))
    action_space = [
      'idle','left','top_left','top','top_right','right',
      'bottom_right','bottom','bottom_left',
      'long_pass','high_pass','short_pass','shot','sprint',
      'release_direction','release_sprint','sliding','dribble','release_dribble'
    ]
    sel_action = action_space.index(str(action[0]))
    formatted_obs_arr = flat_obs + [sel_action]
    return formatted_obs_arr


  def _format_obs(self,obs,action):
    
    obs_copy = copy.deepcopy(obs)
    obs_copy.pop('frame')
    
    ## left team attacking 
    lt_des_player = obs_copy['left_team_designated_player']
    lt_dp_velx, lt_dp_vely = obs_copy['left_team_direction'][lt_des_player][0], obs_copy['left_team_direction'][lt_des_player][1]
    lt_dp_posx, lt_dp_posy = obs_copy['left_team'][lt_des_player][0], obs_copy['left_team'][lt_des_player][1]

    lt_dp_sticky_actions = [i for i in obs_copy['left_agent_sticky_actions'][0]]

    #goal_pos = [1,0]
    #dist_to_goal = [ player_w_ball_pos[i] - goal_pos[i] for i in range(2)]

    ## ball 
    ball_posx, ball_posy, ball_posz = obs_copy['ball'][0], obs_copy['ball'][1], obs_copy['ball'][2]
    ball_vel_x, ball_vel_y = obs_copy['ball_direction'][0], obs_copy['ball_direction'][1]
    ball_rot_x, ball_rot_y, ball_rot_z = obs_copy['ball_rotation'][0], obs_copy['ball_rotation'][1], obs_copy['ball_rotation'][2]

    ## right team goalkeeper
    rt_gk_idx = np.where(obs_copy['right_team_roles'] == 0)
    assert(len(rt_gk_idx) == 1)
    gk_idx = rt_gk_idx[0][0]
    rt_gk_velx, rt_gk_vely = obs_copy['right_team_direction'][gk_idx][0], obs_copy['right_team_direction'][gk_idx][1]
    rt_gk_posx, rt_gk_posy = obs_copy['right_team'][gk_idx][0], obs_copy['right_team'][gk_idx][1]

    action_space = [
      'idle','left','top_left','top','top_right','right',
      'bottom_right','bottom','bottom_left',
      'long_pass','high_pass','short_pass','shot','sprint',
      'release_direction','release_sprint','sliding','dribble','release_dribble'
    ]
    sel_action = action_space.index(str(action[0]))

    pos_obs = [
      lt_dp_posx, lt_dp_posy,
      lt_dp_velx, lt_dp_vely,
      ball_posx, ball_posy, ball_posz,
      ball_vel_x, ball_vel_y,
      ball_rot_x, ball_rot_y, ball_rot_z,
      rt_gk_posx, rt_gk_posy,
      rt_gk_velx,rt_gk_vely,
    ]
    #formatted_obs_arr = pos_obs + lt_dp_sticky_actions + [sel_action]
    formatted_obs_arr = pos_obs + [sel_action]
    return formatted_obs_arr

  def _infer_from_supervised_lr_models(self, x):
    actions_enum = football_action_set.get_action_set(self._config)
    x = x[:-1]
    cs = [c.predict_proba([x])[0][1] for c in self._trained_supervised_lr_models]
    mc = cs.index(max(cs)) 
    #print('infer action --> ',actions_enum[mc] )
    return [actions_enum[mc]]

  def _infer_from_neural_network_model(self, x):
    x = np.array(x)
    x = np.expand_dims(x, axis=1)
    actions_enum = football_action_set.get_action_set(self._config)
    x = x[:-1]
    #model = self._trained_neural_network_model
    model = self._trained_cnn_timeseries_model
    cs = model.predict_classes(np.array([x,]))
    return [actions_enum[cs[0]]]


  def _get_actions(self):
    obs = self._env.observation()

    #print(obs)
    left_actions = []
    right_actions = []
    left_player_position = 0
    right_player_position = 0
    for player in self._players:
      adopted_obs = self._convert_observations(obs, player,
                                               left_player_position,
                                               right_player_position)
      left_player_position += player.num_controlled_left_players()
      right_player_position += player.num_controlled_right_players()
      a = self._action_to_list(player.take_action(adopted_obs))
      assert len(adopted_obs) == len(
          a), 'Player provided {} actions instead of {}.'.format(
              len(a), len(adopted_obs))
      if not player.can_play_right():
        for x in range(player.num_controlled_right_players()):
          index = x + player.num_controlled_left_players()
          a[index] = observation_rotation.flip_single_action(
              a[index], self._config)
      left_actions.extend(a[:player.num_controlled_left_players()])
      ## LukeM: format observation 
      #print('obs ', obs)
      formatted_obs_arr = self._format_rnn_obs(obs,left_actions)
      #formatted_obs_arr = self._format_obs(obs,left_actions)
      ## LukeM: infer action from logistic regression models
      #luke_ai_actions = self._infer_from_supervised_lr_models(formatted_obs_arr)

      ## LukeM: infer action from neural network
      luke_ai_actions = self._infer_from_neural_network_model(formatted_obs_arr)


      self._log_train_data(formatted_obs_arr)
      right_actions.extend(a[player.num_controlled_left_players():])
    ## use game ai ##
    #print('left_actions ', left_actions)
    #actions = left_actions + right_actions
    ## use inference from models ##
    actions = luke_ai_actions + right_actions
    return actions

  def step(self, action):
    action = self._action_to_list(action)
    if self._agent:
      self._agent.set_action(action)
    else:
      assert len(
          action
      ) == 0, 'step() received {} actions, but no agent is playing.'.format(
          len(action))

    _, reward, done, info = self._env.step(self._get_actions())
    score_reward = reward
    if self._agent:
      reward = ([reward] * self._agent.num_controlled_left_players() +
                [-reward] * self._agent.num_controlled_right_players())
    self._cached_observation = None
    info['score_reward'] = score_reward
    return (self.observation(), np.array(reward, dtype=np.float32), done, info)

  def reset(self):
    self._supervised_log_filename = "episode" + str(datetime.datetime.now())

    logfile = open("supervised_logs/"+self._supervised_log_filename,"a")
    sticky_labels = ['action_left','action_top_left','action_top','action_top_right','action_right','action_bottom_right','action_bottom','action_bottom_left','action_sprint','action_dribble']
    cols = ['lt_dp_posx','lt_dp_posy','lt_dp_velx','lt_dp_vely','ball_posx','ball_posy','ball_posz','ball_vel_x','ball_vel_y','ball_rot_x','ball_rot_y','ball_rot_z','rt_gk_posx','rt_gk_posy','rt_gk_velx','rt_gk_vely']
    header = ','.join(cols+sticky_labels+['sel_action'])
    logfile.write(header)
    logfile.write('\n')
    logfile.close()


    self._env.reset()
    for player in self._players:
      player.reset()
    self._cached_observation = None
    return self.observation()

  def observation(self):
    if not self._cached_observation:
      self._cached_observation = self._env.observation()
      if self._agent:
        self._cached_observation = self._convert_observations(
            self._cached_observation, self._agent,
            self._agent_left_position, self._agent_right_position)
    return self._cached_observation

  def write_dump(self, name):
    return self._env.write_dump(name)

  def close(self):
    self._env.close()

  def get_state(self, to_pickle={}):
    return self._env.get_state(to_pickle)

  def set_state(self, state):
    self._cached_observation = None
    return self._env.set_state(state)

  def tracker_setup(self, start, end):
    self._env.tracker_setup(start, end)

  def render(self, mode='human'):
    self._cached_observation = None
    return self._env.render(mode=mode)

  def disable_render(self):
    self._cached_observation = None
    return self._env.disable_render()
