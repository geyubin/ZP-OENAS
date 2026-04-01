import numpy as np
import random


class mesh_crowd(object):
    def __init__(self, curr_archiving_in, curr_archiving_fit, mesh_div, min_, max_, particals):
        self.curr_archiving_in = np.array(curr_archiving_in)
        self.curr_archiving_fit = np.array(curr_archiving_fit)
        self.mesh_div = mesh_div

        self.num_ = self.curr_archiving_in.shape[0]

        self.particals = particals

        self.id_archiving = np.zeros(self.num_)
        self.crowd_archiving = np.zeros(self.num_)

        self.gbest_in = np.zeros((self.particals, self.curr_archiving_in.shape[1]))
        self.gbest_fit = np.zeros((self.particals, self.curr_archiving_fit.shape[1]))
        self.min_ = min_
        self.max_ = max_

    def cal_mesh_id(self, in_):
        id_ = 0
        for i in range(self.curr_archiving_in.shape[1]):
            id_dim = int((in_[i] - self.min_[i]) * self.mesh_div / (self.max_[i] - self.min_[i]))  # self.num_
            id_ = id_ + id_dim * (self.mesh_div ** i)
        return id_

    def divide_archiving(self):
        for i in range(self.num_):
            self.id_archiving[i] = self.cal_mesh_id(self.curr_archiving_in[i])

    def get_crowd(self):
        index_ = (np.linspace(0, self.num_ - 1, self.num_)).tolist()
        index_ = map(int, index_)
        index_ = list(index_)
        while len(index_) > 0:
            index_same = [index_[0]]
            for i in range(1, len(index_)):
                if self.id_archiving[index_[0]] == self.id_archiving[index_[i]]:
                    index_same.append(index_[i])
            number_ = len(index_same)
            for i in index_same:
                self.crowd_archiving[i] = number_
                index_.remove(i)


class get_gbest(mesh_crowd):
    def __init__(self, curr_archiving_in, curr_archiving_fit, mesh_div_num, min_, max_, particals):
        super(get_gbest, self).__init__(curr_archiving_in, curr_archiving_fit, mesh_div_num, min_, max_, particals)
        self.probability_archiving = None
        self.divide_archiving()
        self.get_crowd()

    def get_probability(self):
        # for i in range(self.num_):
        #     self.probability_archiving = 1.0 / (self.crowd_archiving ** 3)
        self.probability_archiving = 1.0 / (self.crowd_archiving ** 3)
        self.probability_archiving = self.probability_archiving / np.sum(self.probability_archiving)

    def get_gbest_index(self):
        random_pro = random.uniform(0.0, 1.0)
        for i in range(self.num_):
            if random_pro <= np.sum(self.probability_archiving[0:i + 1]):
                return i

    def get_gbest(self):
        self.get_probability()
        for i in range(self.particals):
            gbest_index = self.get_gbest_index()
            self.gbest_in[i] = self.curr_archiving_in[gbest_index]
            self.gbest_fit[i] = self.curr_archiving_fit[gbest_index]
        return self.gbest_in, self.gbest_fit


class divide_archiving(get_gbest):
    def __init__(self, curr_archiving_in, curr_archiving_fit, mesh_div_num, min_, max_, particals):
        super(divide_archiving, self).__init__(curr_archiving_in, curr_archiving_fit, mesh_div_num, min_, max_,
                                               particals)
        if self.curr_archiving_in.shape[0] == 0:
            raise ValueError("curr_archiving_in is empty")
        self.divide_archiving()
        self.get_crowd()  # self.crowd_archiving

    def get_dual_archiving(self, use_dual):
        global positive_archiving, negative_archiving
        self.get_probability()
        if self.probability_archiving.shape[0] == 0:
            raise ValueError("probability_archiving is empty")

        sorted_indices = np.argsort(self.probability_archiving)
        sorted_data = [self.curr_archiving_in[i] for i in sorted_indices]
        half_size = len(self.probability_archiving) // 2
        indic = 0 if len(self.probability_archiving) < 2 else 1
        if use_dual:
            if indic == 0:
                positive_archiving = negative_archiving = self.curr_archiving_in
            elif indic == 1:
                positive_archiving = np.array(sorted_data[:half_size])
                negative_archiving = np.array(sorted_data[half_size:])
        else:  # 单存档（消融实验）
            positive_archiving = negative_archiving = self.curr_archiving_in

        if positive_archiving.size == 0 or negative_archiving.size == 0:
            raise ValueError("(in archive.py) Either positive or negative archiving is empty")
        in_pgbest = random.choice(positive_archiving)
        in_ngbest = random.choice(negative_archiving)
        return in_pgbest, in_ngbest, indic


class clear_archiving(get_gbest):
    def __init__(self, curr_archiving_in, curr_archiving_fit, mesh_div_num, min_, max_, particals):
        super(get_gbest, self).__init__(curr_archiving_in, curr_archiving_fit, mesh_div_num, min_, max_, particals)
        self.divide_archiving()
        self.get_crowd()

    def get_probability(self):
        # for i in range(self.num_):
        #     self.probability_archiving = self.crowd_archiving / sum(self.crowd_archiving)
        self.probability_archiving = self.crowd_archiving / sum(self.crowd_archiving)

    def get_clear_index(self):
        len_clear = self.curr_archiving_in.shape[0] - self.thresh
        clear_index = []
        while len(clear_index) < len_clear:
            random_pro = random.uniform(0.0, np.sum(self.probability_archiving))
            for i in range(self.num_):
                if random_pro <= np.sum(self.probability_archiving[0:i + 1]):
                    if i not in clear_index:
                        clear_index.append(i)
                        break
        return clear_index

    def clear_(self, thresh):
        self.thresh = thresh
        # self.archiving_size = archiving_size
        self.get_probability()
        clear_index = self.get_clear_index()
        # gbest_index=self.get_gbest_index()
        self.curr_archiving_in = np.delete(self.curr_archiving_in, clear_index, axis=0)
        self.curr_archiving_fit = np.delete(self.curr_archiving_fit, clear_index, axis=0)
        return self.curr_archiving_in, self.curr_archiving_fit
