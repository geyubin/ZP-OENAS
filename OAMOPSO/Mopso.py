
from fitness_funs import *
import init
import update

import archive


class Mopso:
    def __init__(self, particals, w, c1, c2, max_, min_, thresh, input_train, input_valid, mesh_div=10, device='cuda:7',
                 constraint=None, use_elite_init=True, use_dual=True, use_elite_update=True):
        self.w, self.c1, self.c2 = w, c1, c2
        self.mesh_div = mesh_div
        self.particals = particals
        self.thresh = thresh
        self.max_ = max_
        self.min_ = min_
        self.max_v = (max_ - min_) * 0.05
        self.min_v = (max_ - min_) * 0.05 * (-1)
        self.input_train = input_train
        self.input_valid = input_valid
        self.indic_ = [0]
        self.difference = []
        self.device = device
        self.constraint = constraint
        self.use_elite_init = use_elite_init
        self.use_dual = use_dual
        self.use_elite_update = use_elite_update

    def evaluation_fitness(self, in_):
        fitness_curr = []
        for i in range(len(in_)):
            fitness_curr.append(
                fitness_(in_[i], self.input_train, self.input_valid, self.indic_, self.device, self.min_, self.max_,
                         self.constraint, self.use_elite_update))
        self.fitness_ = np.array(fitness_curr)

    def initialize(self):
        if self.use_elite_init:
            self.in_temp = init.init_designparams(self.particals, self.min_, self.max_)
            self.v_temp = init.init_v(self.particals, self.min_v, self.max_v)

            self.evaluation_fitness(self.in_temp)
            ALC_init = [item[1] for item in self.fitness_]
            sorted_idx = np.argsort(ALC_init)
            half_size = len(ALC_init) // 2
            high_quality = sorted_idx[:half_size // 2 + 1]
            self.in_, self.v_ = [], []
            for idx in high_quality:
                self.in_.append(self.in_temp[idx, :])
                self.v_.append(self.v_temp[idx, :])

            low_quality = sorted_idx[half_size + 1:]
            if len(low_quality) > 0 and half_size // 2 > 0:
                random_rows = np.random.choice(low_quality, min(half_size // 2, len(low_quality)), replace=False)
                for idx in random_rows:
                    self.in_.append(self.in_temp[idx, :])
                    self.v_.append(self.v_temp[idx, :])

            self.in_ = np.array(self.in_)
            self.v_ = np.array(self.v_)
        else:
            self.in_ = init.init_designparams(self.particals, self.min_, self.max_)
            self.v_ = init.init_v(self.particals, self.min_v, self.max_v)

        self.evaluation_fitness(self.in_)

        self.in_p, self.fitness_p = init.init_pbest(self.in_, self.fitness_)
        self.archive_in, self.archive_fitness = init.init_archive(self.in_, self.fitness_)
        self.in_g, self.fitness_g = init.init_gbest(self.archive_in, self.archive_fitness, self.mesh_div, self.min_,
                                                    self.max_, self.particals)

    def update_(self, t):
        dual_archiving = archive.divide_archiving(self.archive_in, self.archive_fitness, self.mesh_div, self.min_,
                                                  self.max_, self.particals)
        in_pgbest, in_ngbest, indic = dual_archiving.get_dual_archiving(self.use_dual)

        if self.use_dual == True:
            if indic == 1:
                self.v_ = update.update_v_1(self.v_, self.min_v, self.max_v, self.in_, self.in_p, in_pgbest, in_ngbest,
                                            self.w, self.c1, self.c2, t)
            elif indic == 0:
                self.v_ = update.update_v_0(self.v_, self.min_v, self.max_v, self.in_, self.in_p, in_pgbest,
                                            self.w, self.c1, self.c2)
            self.in_ = update.update_in(self.in_, self.v_, self.min_, self.max_)
        else:
            self.v_ = update.update_v_0(self.v_, self.min_v, self.max_v, self.in_, self.in_p, in_pgbest,
                                        self.w, self.c1, self.c2)
            self.in_ = update.update_in(self.in_, self.v_, self.min_, self.max_)

        self.evaluation_fitness(self.in_)
        self.in_p, self.fitness_p = update.update_pbest(self.in_, self.fitness_, self.in_p, self.fitness_p)
        self.archive_in, self.archive_fitness, difference_temp = update.update_archive(self.in_, self.fitness_,
                                                                                       self.archive_in,
                                                                                       self.archive_fitness,
                                                                                       self.thresh, self.mesh_div,
                                                                                       self.min_, self.max_,
                                                                                       self.particals, self.indic_,
                                                                                       self.device,
                                                                                       self.input_train,
                                                                                       self.input_valid)

        self.difference.append(difference_temp)
        if t >= 2:
            delta = self.difference[-1] - max(self.difference[1:-1])
            indic_ = 1 if delta > 0 else 0
            self.indic_.append(indic_)

    def done(self, cycle_):
        self.initialize()
        max_memory_history = []
        for i in range(cycle_):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            self.update_(t=i)

        return self.archive_in, self.archive_fitness
