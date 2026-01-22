import os
import subprocess

# NOTE: edit this
h1_test_args = [
    {'task': 'h1_task_reach', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_task_reach', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_task_button', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_task_button', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_task_cabinet', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_task_cabinet', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_task_ball', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_task_ball', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_task_box', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_task_box', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_task_lift', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_task_lift', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_task_transfer', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_task_transfer', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_task_carry', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_task_carry', 'load_run': '1000_ppo', 'checkpoint': -1},
]
g1_test_args = [
    {'task': 'g1_task_reach', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'g1_task_reach', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'g1_task_button', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'g1_task_button', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'g1_task_cabinet', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'g1_task_cabinet', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'g1_task_ball', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'g1_task_ball', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'g1_task_box', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'g1_task_box', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'g1_task_lift', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'g1_task_lift', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'g1_task_transfer', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'g1_task_transfer', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'g1_task_carry', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'g1_task_carry', 'load_run': '1000_ppo', 'checkpoint': -1},
]
h1_2_test_args = [
    {'task': 'h1_2_task_reach', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_2_task_reach', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_2_task_button', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_2_task_button', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_2_task_cabinet', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_2_task_cabinet', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_2_task_ball', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_2_task_ball', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_2_task_box', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_2_task_box', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_2_task_lift', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_2_task_lift', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_2_task_transfer', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_2_task_transfer', 'load_run': '1000_ppo', 'checkpoint': -1},
    {'task': 'h1_2_task_carry', 'load_run': '0000_best', 'checkpoint': -1},
    {'task': 'h1_2_task_carry', 'load_run': '1000_ppo', 'checkpoint': -1},
]

test_args = h1_2_test_args + g1_test_args + h1_test_args
gpu_id = 0

exception_list = []

for arg_dict in test_args:
    try:
        task = arg_dict['task']
        experiment_name = task
        run_name = arg_dict['load_run']
        checkpoint = arg_dict['checkpoint']
        try:
            subprocess.run(f'python legged_gym/scripts/evaluate.py --task {task} --experiment_name {experiment_name} --load_run {run_name} --checkpoint {checkpoint} --sim_device cuda:{gpu_id} --rl_device cuda:{gpu_id} --visualize', shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(e)
            exception_list.append(f"{task}_{run_name}")
    except KeyboardInterrupt:
        print("Interrupted by user")
        break
    
print('=====> Exceptions:', exception_list)