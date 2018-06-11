import multiprocessing
import subprocess

def work(cmd):
    return subprocess.call(cmd, shell=False)

if __name__ == '__main__':
    suffix = '151'
    N_grid = [str(500)]*12 + [str(300)]*12
    num_try_grid = [str(i) for i in range(12)]*2
    parameters = [' '.join((str(i%2), N_grid[i], num_try_grid[i], suffix)) for i in range(24)]
    print('Creating pool')
    pool = multiprocessing.Pool(processes=12)
    print('Starting experiment')
    pool.map(work, [' '.join(('python run_optimization_mnist.py', par)).split(' ') for par in parameters])
