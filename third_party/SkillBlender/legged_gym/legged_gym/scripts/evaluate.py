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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import math

import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, set_seed
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

from legged_gym.scripts.play import H, W, ROBOT_LIST, WALKING, REACHING, STEPPING, SQUATTING, TASK_REACH, TASK_BOX, TASK_BUTTON, TASK_LIFT, TASK_BALL, TASK_CARRY, TASK_TRANSFER, TASK_CABINET
from legged_gym.scripts.play import visualize_task
    
def override_env_cfg(env_cfg: LeggedRobotCfg, args):
    print('====> URDF file:', env_cfg.asset.file)
    # override some parameters for testing
    default_num_envs = 1
    if args.task in WALKING:
        env_cfg.env.episode_length_s = 24
    elif args.task in TASK_REACH:
        env_cfg.env.episode_length_s = 20 * 100
        env_cfg.human.freq = 2
    elif args.task in TASK_BOX+TASK_TRANSFER+TASK_BUTTON:
        env_cfg.env.episode_length_s = 2.5
    elif args.task in TASK_LIFT+TASK_BALL+TASK_CARRY+TASK_CABINET:
        env_cfg.env.episode_length_s = 2.5
    else:
        env_cfg.env.episode_length_s = 8
    env_cfg.env.num_envs = default_num_envs
    # env_cfg.env.mesh_type = "plane"
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    # env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = False
    # env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # env_cfg.commands.ranges.l_wrist_pos_x = [-0.5, 0.5]
    # env_cfg.commands.ranges.l_wrist_pos_y = [-0.5, 0.5]
    # env_cfg.commands.ranges.l_wrist_pos_z = [-0.5, 0.5]
    # env_cfg.commands.ranges.r_wrist_pos_x = [-0.5, 0.5]
    # env_cfg.commands.ranges.r_wrist_pos_y = [-0.5, 0.5]
    # env_cfg.commands.ranges.r_wrist_pos_z = [-0.5, 0.5]
    # env_cfg.commands.ranges.wrist_max_radius = 0.5
    
    return env_cfg

def get_camera_pose(task):
    if not EGO_CENTRIC:
        if task in TASK_BUTTON+TASK_BALL+TASK_CABINET:
            camera_offset = gymapi.Vec3(-1, -2, 1)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                        np.deg2rad(45))
        else:
            camera_offset = gymapi.Vec3(1, -1, 1)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
    else:
        camera_offset = gymapi.Vec3(0.1, 0, 0.9)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0),
                                                    np.deg2rad(45))
    return gymapi.Transform(camera_offset, camera_rotation)

def evaluate(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task, load_run=args.load_run, experiment_name=args.experiment_name)
    HRL = "_task_" in args.task and "Hierarchical" in train_cfg.runner.policy_class_name
    RESET_FLAG = False
    
    env_cfg = override_env_cfg(env_cfg, args)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.run_name = 'evaluate'

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device, hrl=HRL)
    
    model_name = f'{args.load_run}_{train_cfg.runner.resume_path.split("_")[-1].split(".")[0]}'
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'evaluation', 'policies')
        filename_pt = f'{model_name}.pt'
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, filename_pt)
        print('Exported policy as jit script to: ', path)

    robot_index = 0 # which robot (env) is used for logging
    state_log_interval = 1000 # number of steps before plotting states
    rew_log_interval = env.max_episode_length - 1 # number of steps before print average episode rewards
    frame_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'evaluation', 'frames')
    if RECORD_FRAMES:
        os.makedirs(frame_path, exist_ok=True)
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = W
        camera_properties.height = H
        cam = env.gym.create_camera_sensor(env.envs[robot_index], camera_properties)
        camera_pose = get_camera_pose(args.task)
        actor_handle = env.gym.get_actor_handle(env.envs[robot_index], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[robot_index], actor_handle, 0)
        env.gym.attach_camera_to_body(
            cam, 
            env.envs[robot_index], 
            body_handle,
            camera_pose,
            gymapi.FOLLOW_POSITION if not EGO_CENTRIC else gymapi.FOLLOW_TRANSFORM
        )

    max_steps = int(env.max_episode_length) * 100 # sufficient time for 10 rollouts
    done_rollouts = 0
    total_rollouts = 20
    
    if args.task in REACHING+TASK_REACH:
        last_ref_wrist_pos = env.ref_wrist_pos[robot_index][:,:3].cpu().numpy()
    if args.task in STEPPING:
        last_ref_feet_pos = env.ref_feet_pos[robot_index][:,:2].cpu().numpy()
    if args.task in SQUATTING:
        last_ref_root_height = env.ref_root_height[robot_index].cpu().numpy()

    logger = Logger(env.dt)
    env.ori_root_states = env.root_states.clone()
    if RECORD_FRAMES:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        filename_mp4 = f'{args.task}_{model_name}.mp4'
        video = cv2.VideoWriter(os.path.join(frame_path, filename_mp4), fourcc, 25.0, (W, H))
    for i in tqdm(range(max_steps)):
        if done_rollouts >= total_rollouts:
            break
        
        if RECORD_FRAMES:
            # visualize task-related information
            visualize_task(args.task, env)

        actions = policy(obs.detach())
        actions = actions["actions_mean"] if HRL else actions

        if FIX_COMMAND:
            env.commands[:, 0] = 1.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.0

        obs, _, rews, dones, infos = env.step(actions.detach())
        if dones[robot_index] and args.task not in REACHING+TASK_REACH+STEPPING+SQUATTING:
            done_rollouts += 1

        if (args.task == 'g1_task_cabinet' or args.task == 'h1_2_task_cabinet') and i <= 1000:
            continue # still need to fix accuracy, but all methods are 0.000 so it's okay
        
        if RECORD_FRAMES:
            if i % 4 == 0:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(env.sim, env.envs[robot_index], cam, gymapi.IMAGE_COLOR)
                img = np.reshape(img, (H, W, 4))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video.write(img[..., :3])
            
        logger.log_states(
            {
                'dof_pos_target': actions[robot_index, :].detach().cpu().numpy() * env.cfg.control.action_scale,
                'dof_pos': env.dof_pos[robot_index, :].detach().cpu().numpy(),
                'dof_vel': env.dof_vel[robot_index, :].detach().cpu().numpy(),
                'dof_torque': env.torques[robot_index, :].detach().cpu().numpy(),
                'command_x': env.commands[robot_index, 0].detach().cpu().numpy(),
                'command_y': env.commands[robot_index, 1].detach().cpu().numpy(),
                'command_yaw': env.commands[robot_index, 2].detach().cpu().numpy(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].detach().cpu().numpy(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].detach().cpu().numpy(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].detach().cpu().numpy(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].detach().cpu().numpy(),
                'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].detach().cpu().numpy(),
                'base_roll': env.base_euler_xyz[robot_index, 0].detach().cpu().numpy(),
                'base_pitch': env.base_euler_xyz[robot_index, 1].detach().cpu().numpy(),
                'base_height': env.root_states[robot_index, 2].detach().cpu().numpy(),
            }
        )

        if args.task in REACHING+TASK_REACH:
            ref_wrist_pos = env.ref_wrist_pos[robot_index][:,:3].cpu().numpy()
            wrist_pos = env.rigid_state[robot_index, env.wrist_indices, :3].cpu().numpy()
            
            # if wrist_pos updates, which means that ref_wrist_pos is not last_wrist_pos
            if dones[robot_index]:
                print("=> Accidentally reset")
                RESET_FLAG = True
            if not np.allclose(ref_wrist_pos, last_ref_wrist_pos):
                if RESET_FLAG:
                    RESET_FLAG = False
                    last_ref_wrist_pos = ref_wrist_pos
                else:
                    done_rollouts += 1
                    print("=> update reach goal")
                    wrist_pos_diff = np.mean(np.abs(last_ref_wrist_pos - wrist_pos))
                    logger.log_metrics(
                        {
                            'wrist_pos_diff': wrist_pos_diff
                        }
                    )
                    last_ref_wrist_pos = ref_wrist_pos
                    print('wrist_pos_diff:', wrist_pos_diff)
                
        if args.task in STEPPING:
            ref_feet_pos = env.ref_feet_pos[robot_index][:,:2].cpu().numpy()
            feet_pos = env.rigid_state[robot_index, env.feet_indices, :2].cpu().numpy()
            
            # if feet_pos updates, which means that ref_feet_pos is not last_feet_pos
            if dones[robot_index]:
                print("=> Accidentally reset")
                RESET_FLAG = True
            if not np.allclose(ref_feet_pos, last_ref_feet_pos):
                if RESET_FLAG:
                    RESET_FLAG = False
                    last_ref_feet_pos = ref_feet_pos
                else:
                    done_rollouts += 1
                    print("=> update stepping goal")
                    feet_pos_diff = np.mean(np.abs(last_ref_feet_pos - feet_pos))
                    logger.log_metrics(
                        {
                            'feet_pos_diff': feet_pos_diff
                        }
                    )
                    last_ref_feet_pos = ref_feet_pos
                    print('feet_pos_diff:', feet_pos_diff)
        
        if args.task in SQUATTING:
            ref_root_height = env.ref_root_height[robot_index].cpu().numpy()
            root_height = env.root_states[robot_index, 2].cpu().numpy()
            
            # if root_height updates, which means that ref_root_height is not last_root_height
            if not np.allclose(ref_root_height, last_ref_root_height):
                done_rollouts += 1
                print("=> update squatting goal")
                root_height_diff = np.mean(np.abs(last_ref_root_height - root_height))
                logger.log_metrics(
                    {
                        'root_height_diff': root_height_diff
                    }
                )
                last_ref_root_height = ref_root_height
                print('root_height_diff:', root_height_diff)
                
        if args.task in TASK_BOX+TASK_TRANSFER:
            if not dones[robot_index]:
                box_pos = env.box_root_states[robot_index, :3].cpu().numpy()
                box_goal_pos = env.box_goal_pos[robot_index, :3].cpu().numpy()
                box_pos_diff = np.mean(np.abs(box_pos - box_goal_pos))
            if dones[robot_index]:
                logger.log_metrics(
                    {
                        'box_pos_diff': box_pos_diff
                    }
                )
                print('box_pos_diff:', box_pos_diff)
                
        if args.task in TASK_BUTTON:
            if not dones[robot_index]:
                button_goal_pos = env.button_goal_pos[robot_index, 1:3].cpu().numpy()
                wrist_pos = env.rigid_state[robot_index, env.wrist_indices, 1:3].cpu().numpy()
                wrist_pos = wrist_pos[0] # left hand
                wrist_pos_diff = np.mean(np.abs(button_goal_pos - wrist_pos))
            if dones[robot_index]:
                logger.log_metrics(
                    {
                        'button_pos_diff': wrist_pos_diff
                    }
                )
                print('button_pos_diff:', wrist_pos_diff)
                
        if args.task in TASK_LIFT:
            if not dones[robot_index]:
                box_goal_pos = env.box_goal_pos[robot_index].cpu().numpy()
                box_pos = env.box_root_states[robot_index, :3].cpu().numpy()
                box_pos_diff = np.mean(np.abs(box_pos - box_goal_pos)[2:3]) # z axis only
            if dones[robot_index]:
                if done_rollouts == 1:
                    total_rollouts += 1 # skip the first rollout due to reset errors
                else:
                    logger.log_metrics(
                        {
                            'box_pos_diff': box_pos_diff
                        }
                    )
                    print('box_pos_diff:', box_pos_diff)
        
        if args.task in TASK_BALL:
            if not dones[robot_index]:
                ball_goal_pos = env.goal_pos[robot_index].cpu().numpy()
                ball_pos = env.ball_root_states[robot_index, :3].cpu().numpy()
                ball_pos_diff = np.mean(np.abs(ball_pos - ball_goal_pos))
            if dones[robot_index]:
                logger.log_metrics(
                    {
                        'ball_pos_diff': ball_pos_diff
                    }
                )
                print('ball_pos_diff:', ball_pos_diff)
        
        if args.task in TASK_CARRY:
            if not dones[robot_index]:
                box_goal_pos = env.box_goal_pos[robot_index].cpu().numpy()
                box_pos = env.box_root_states[robot_index, :3].cpu().numpy()
                box_pos_diff = np.mean(np.abs(box_pos - box_goal_pos))
            if dones[robot_index]:
                if done_rollouts == 1:
                    total_rollouts += 1 # skip the first rollout due to reset errors
                else:
                    logger.log_metrics(
                        {
                            'box_pos_diff': box_pos_diff
                        }
                    )
                    print('box_pos_diff:', box_pos_diff)
        
        if args.task in TASK_CABINET:
            if not dones[robot_index]:
                arti_obj_dof_diff = (env.arti_obj_dof_state[:, :, 0] - env.arti_obj_dof_goal).cpu().numpy() # [num_envs, 2]
                arti_obj_dof_diff = np.mean(np.abs(arti_obj_dof_diff))
            if dones[robot_index]:
                logger.log_metrics(
                    {
                        'arti_obj_dof_diff': arti_obj_dof_diff
                    }
                )
                print('arti_obj_dof_diff:', arti_obj_dof_diff)
            
        ### logging end ###

        # if i > 0 and i % state_log_interval == 0:
        #     logger.plot_states()

        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)
        if i > 0 and i % rew_log_interval == 0:
            logger.print_rewards()

    fig = logger._plot(show=False)
    filename_png = f'{model_name}.png'
    fig.savefig(os.path.join(frame_path, filename_png))
    if RECORD_FRAMES:
        video.release()
        print('===> Video saved to:', os.path.join(frame_path, filename_mp4))
    del logger

if __name__ == '__main__':
    EXPORT_POLICY = True
    EGO_CENTRIC = False
    FIX_COMMAND = True
    args = get_args(test=True)
    RECORD_FRAMES = args.visualize # edit this in run_evaluation.py
    args.headless = not args.visualize
    evaluate(args)