from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import pandas as pd
from pathlib import Path
import csv
import warnings
import networkx as nx
import itertools
warnings.filterwarnings('ignore')

from Generators.maxcut_benchmark_generator import get_benchmark_graph
from Solvers.Dwave_aneal import DwaveAnneal
from Diplom.Mk_8_1 import LargeScaleRQAOA_mk8v1
from Diplom.Bench_mod import mod


class Benchmark:

    def __init__(self,
                 solver,
                 problem,
                 configs,
                 saving_path,
                 filename='Benchmark'):


        self.solver = solver
        self.problem = problem
        self.configs = configs
        self.saving_path = Path(saving_path)
        self.filename = filename

        self.param_grid = {}
        self.keys = []
        self.path = ''
        self.columns = []
        self.counter = 0


    def file_preparing(self):

        self.path = self.saving_path / (self.filename + '.csv')
        self.columns = ['task'] + self.keys + self.configs['collected_data']

        try:
            with open(self.path, "r") as file:
                for line in tqdm(file.readlines()[1:], desc='Reading file'):
                    try:
                        split_line = line.replace("\n", "").split(",")
                        self.param_grid.pop(split_line[0])
                        self.counter += 1
                    except (IndexError, KeyError):
                        continue

        except FileNotFoundError:
            print('Creating file...')
            data = pd.DataFrame(columns=self.columns)
            # noinspection PyTypeChecker
            data.to_csv(self.path,
                        index=False,
                        header=True,
                        quoting=csv.QUOTE_ALL,
                        sep=',',
                        escapechar='\\',
                        lineterminator='\n',
                        encoding='utf-8')

    def run_single_benchmark(self, params):

        try:

            problem_instance = nx.Graph(self.problem(**params['problem_config']))
            solver_instance = self.solver(graph=problem_instance.copy(),
                                          **params['solver_config'])
            solver_instance.solve()

            results = [str(getattr(solver_instance, att)) for att in self.configs['collected_data']]
            problem_params = list(params['problem_config'].values())
            solver_params = list(params['solver_config'].values())
            task_params = [str(p) for p in (problem_params + solver_params)]

            row_data = params['task'] + task_params + results

            with open(self.path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file,
                                    quoting=csv.QUOTE_MINIMAL,
                                    escapechar='\\',
                                    lineterminator='\n')
                writer.writerow(row_data)

        except Exception as e:
            print("\033[33m {}".format(f"==" * 30))
            print('Exception occurred')
            print(f'Problem config : {params['problem_config']}')
            print(f'Solver config : {params['solver_config']}')
            print(e)
            print("\033[33m {}".format(f"==" * 30))

    def run_benchmark(self, n_jobs=1):

        self.generate_configs_grid()
        self.file_preparing()

        verbose = 0
        with tqdm_joblib(tqdm(desc="Benchmarking",
                              total=len(self.param_grid),
                              position=0,
                              leave=True)) as progress_bar:
            results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(self.run_single_benchmark)(params) for params in self.param_grid.values())

    def generate_configs_grid(self):

        params = []
        problem_keys = list(self.configs['problem_config'].keys())
        problems_values = list(self.configs['problem_config'].values())

        solver_keys = list(self.configs['solver_config'].keys())
        solver_values = list(self.configs['solver_config'].values())

        self.keys = problem_keys + solver_keys
        values = problems_values + solver_values

        for combination in itertools.product(*values):
            params.append(dict(zip(self.keys, combination)))

        self.param_grid = {str(task):  {'task': [str(task)],
                                        'problem_config': {x: dct[x] for x in problem_keys},
                                        'solver_config': {x: dct[x] for x in solver_keys}}
                           for task, dct in enumerate(params)}


if __name__ == "__main__":

    param_grid = {
        'problem_config': {
            'seed' : list(range(297)),
        },

        'solver_config': {
            'num_reads' : [1000],
        },

        'collected_data': [
            'energy',

        ]
    }


    bench_1 = Benchmark(solver=DwaveAnneal,
                        problem=get_benchmark_graph,
                        configs=param_grid,
                        saving_path='../Data/Anneal',
                        filename='Benchmark')
    bench_1.run_benchmark(n_jobs=4)









