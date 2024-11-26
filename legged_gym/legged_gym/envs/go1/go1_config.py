# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go1RoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_horizons = 6
        num_envs = 4096
        num_observations = 45 * num_horizons
        num_privileged_obs = 3+45+ 187 + 29 + 3 # lin vel + obs + scandots + privileged randomization + disturbance
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]
             
            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]
             
            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
     
    class commands(LeggedRobotCfg.commands):
        max_curriculum = 1.5
        curriculum = True
        heading_command = True
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1,1]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]
     
    class terrain(LeggedRobotCfg.terrain):
        measure_heights = True
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        num_rows = 10
        num_cols = 20
     
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 28.}  # [N*m/rad]
        damping = {'joint': 0.7}    # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
         
        randomize_base_com = True
        added_com_range = [-0.05, 0.05]
          
        randomize_motor = True
        motor_strength_range = [0.9,1.1]
        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 1

        disturbance = True
        disturbance_range = [-30,30]
        disturbance_internal = 8
     
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf","base"]
        terminate_after_contacts_on = ["base"]
        privileged_contact_on = ["thigh","calf","base"]
        self_collisions = 1 # 1 to disable, 0 to enable... bitwise filter
     
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False
        base_height_target = 0.30
        clearance_height_target = -0.2
        use_early_terminate = False
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
         
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0
            energy = -1e-5
            feet_air_time = 0.0
            foot_clearance = 0
            feet_drag = -0.1
            smoothness = 0
            default_pos = 0
          
        # termination constraint 
        class constraints:
            dof_pos_limits = 1.0 
            dof_vel_limits = 1.0
            dof_torque_limits = 1.0
            action_rate = 1.0
            action_smoothness = 1.0
         
class Go1RoughCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'ConstrainedOnPolicyRunner'
    class policy(LeggedRobotCfgPPO.policy):
        constraints_num = len([attr for attr in Go1RoughCfg.rewards.constraints.__dict__ if not attr.startswith('__')])
        horizon = Go1RoughCfg.env.num_horizons
        measurement = len(Go1RoughCfg.terrain.measured_points_x) * len(Go1RoughCfg.terrain.measured_points_y)
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        constraints_num = len([attr for attr in Go1RoughCfg.rewards.constraints.__dict__ if not attr.startswith('__')])
        initialize_kappa = 0.1
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        num_steps_per_env = 24
        algorithm_class_name = "RACPPO"
        max_iterations = 10000
        policy_class_name = "ActorCriticRAC"
        run_name = ''
        experiment_name = 'go1_rough'
