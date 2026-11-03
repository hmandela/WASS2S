"""Bounded, ordered CPU work for independent MME truncation candidates."""
import os


def tuning_budget(task_count):
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    allocation = min(affinity, int(os.environ.get('SLURM_CPUS_PER_TASK', affinity)))
    requested = int(os.environ.get('WAS_MME_TUNING_JOBS', '1'))
    threads = int(os.environ.get('WAS_MME_TUNING_BLAS_THREADS', '1'))
    if min(allocation, requested, threads) < 1:
        raise ValueError('MME tuning CPU budgets must be positive')
    threads = min(threads, allocation)
    jobs = min(requested, max(1, allocation // threads), max(1, task_count))
    return jobs, threads


def ordered_tuning_map(function, arguments):
    """Thread workers own fresh models; return the original candidate order.

    The caller owns fold-local prepared fields. No global cache, random-state
    reset, process copy of the full dataset, or nested Dask scheduler is used.
    """
    jobs, threads = tuning_budget(len(arguments))
    if jobs == 1:
        return [function(*args) for args in arguments]
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits
    print(f'[MME tuning] {len(arguments)} fits; workers={jobs}; BLAS/worker={threads}', flush=True)
    with threadpool_limits(limits=threads):
        return Parallel(n_jobs=jobs, backend='threading', pre_dispatch=jobs, batch_size=1)(
            delayed(function)(*args) for args in arguments)
