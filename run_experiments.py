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
        env, run_id, div_scale = exp
        args = f"python -m planet.scripts.train --logdir logs/cpc_{env}_divscale{div_scale}/run_{run_id} " + \
               f"--params '{{tasks: [{env}], model: cpcm, divergence_scale: {div_scale}}}'"
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
    div_scales = [10, 100]

    exps = list(itertools.product(envs, run_ids, div_scales))
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
