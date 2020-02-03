import itertools
import argparse
import math
import multiprocessing as mp
import shlex
import subprocess
import os


def worker(gpu_id, max_per_gpu, exps):
    sh_env = os.environ.copy()
    sh_env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    processes = []
    for exp in exps:
        env, run_id, batch_shape = exp
        args = f"python -m planet.scripts.train --logdir logs/planet_ssm_bs{'_'.join(map(str, batch_shape))}/{env}/run_{run_id} " + \
               f"--params '{{tasks: [{env}], model: ssm, batch_shape: {batch_shape}}}'"
        print('Running', args)
        args = shlex.split(args)
        processes.append(subprocess.Popen(args, env=sh_env))

        if len(processes) >= max_per_gpu:
            [p.wait() for p in processes]
            processes = []

    [p.wait() for p in processes]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=4)
    parser.add_argument('--max_per_gpu', type=int, default=4)
    parser.add_argument('--n_runs', type=int, default=2)
    args = parser.parse_args()

    envs = ['cheetah_run', 'cartpole_swingup', 'finger_spin', 'walker_walk']
    run_ids = list(range(args.n_runs))
    batch_shapes = [(128, 2), (3, 50)]

    exps = list(itertools.product(envs, run_ids, batch_shapes))
    print(f'Running {len(exps)} experiments')
    n_exps = len(exps)
    chunk_size = math.ceil(n_exps / args.n_gpus)
    worker_args = []
    for i in range(args.n_gpus):
        start, end = chunk_size * i, min(chunk_size * (i + 1), n_exps)
        worker_args.append((i, args.max_per_gpu, exps[start:end]))
    workers = [mp.Process(target=worker, args=arg) for arg in worker_args]
    [w.start() for w in workers]
    [w.join() for w in workers]
