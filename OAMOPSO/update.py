import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import random

import torch
from sklearn.preprocessing import MinMaxScaler

import pareto
import archive
from EvoNet import SearchSpace

from dataPrepare import build_train_Optimizer_Loss
from metrics import LE


def indicator(x_list, y_list):
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    diff = y_array - x_array
    result = [1 if x >= 0 else 0 for x in diff]
    return result

def vectorized_discretize(in_array, min_vals, max_vals):
    in_array = np.asarray(in_array)
    min_vals = np.asarray(min_vals)
    max_vals = np.asarray(max_vals)

    n_classes = np.round(max_vals - min_vals + 1).astype(int)
    intervals = (max_vals - min_vals) / n_classes

    idx = ((in_array - min_vals) / intervals).astype(int)
    idx = np.clip(idx, 0, n_classes - 1)

    return (min_vals + idx).astype(int)

def update_v_1(v_, v_min, v_max, in_, in_pbest, in_pgbest, in_ngbest, w, c1, c2, t=0):
   
    v_temp = w * v_ + c1 * random.random() * (in_pbest - in_) + c2 * random.random() * (in_pgbest - in_) - np.exp(-t) * (in_ngbest - in_)
   
    for i in range(v_temp.shape[0]):
        for j in range(v_temp.shape[1]):
            if v_temp[i, j] < v_min[j]:
                v_temp[i, j] = v_min[j]
            if v_temp[i, j] > v_max[j]:
                v_temp[i, j] = v_max[j]
    return v_temp


def update_v_0(v_, v_min, v_max, in_, in_pbest, in_gbest, w, c1, c2, t=0):
    
    v_temp = w * v_ + c1 * random.random() * (in_pbest - in_) + c2 * random.random() * (in_gbest - in_)
   
    for i in range(v_temp.shape[0]):
        for j in range(v_temp.shape[1]):
            if v_temp[i, j] < v_min[j]:
                v_temp[i, j] = v_min[j]
            if v_temp[i, j] > v_max[j]:
                v_temp[i, j] = v_max[j]
    return v_temp

def update_in(in_, v_, in_min, in_max):
    
    in_temp = in_ + v_
    
    for i in range(in_temp.shape[0]):
        for j in range(in_temp.shape[1]):
            if in_temp[i, j] < in_min[j]:
                in_temp[i, j] = in_min[j]
            if in_temp[i, j] > in_max[j]:
                in_temp[i, j] = in_max[j]
    return in_temp


def compare_pbest(in_indiv, pbest_indiv):
    num_greater = 0
    num_less = 0
    for i in range(len(in_indiv)):
        if in_indiv[i] < pbest_indiv[i]:
            num_greater = num_greater + 1
        if in_indiv[i] > pbest_indiv[i]:
            num_less = num_less + 1
   
    if num_greater > 0 and num_less == 0:
        return True
    
    elif num_greater == 0 and num_less > 0:
        return False
    else:
        
        random_ = random.uniform(0.0, 1.0)
        if random_ > 0.5:
            return True
        else:
            return False


def update_pbest(in_, fitness_, in_pbest, out_pbest):
    for i in range(out_pbest.shape[0]):
        
        if compare_pbest(fitness_[i], out_pbest[i]):
            out_pbest[i] = fitness_[i]
            in_pbest[i] = in_[i]
    return in_pbest, out_pbest


def update_archive(in_, fitness_, archive_in, archive_fitness, thresh, mesh_div, min_, max_, particals, indic_, device, input_train, input_valid):
    pareto_1 = pareto.Pareto_(in_, fitness_)
    curr_in, curr_fit = pareto_1.pareto()
    if len(indic_) == 1:
        pre_indic = indic = 0
    else:
        pre_indic, indic = indic_[-2], indic_[-1] 

    if pre_indic != indic:
        dataset = getattr(input_train, "dataset", None)
        base_dataset = getattr(dataset, "dataset", dataset)
        if base_dataset is not None and hasattr(base_dataset, "classes"):
            num_classes = len(base_dataset.classes)
        else:
            num_classes = 10

        if indic == 1:
            for i, temp_in in enumerate(archive_in):
                rounded_in = vectorized_discretize(temp_in, min_, max_)

                model = SearchSpace(rounded_in, num_classes=num_classes, device=device, p=0, stem_multiplier=2, init=False).to(device)
                with torch.no_grad():
                    dummy = torch.randn(1, 3, 32, 32).to(device)
                    _ = model(dummy)
                model.train()
                train_criterion, eval_criterion, optimizer, scheduler = build_train_Optimizer_Loss(model, 0.9, 0.025, 3e-4, epochs=8, device=device)

                for epoch in range(8):
                    running_loss = 0.0
                    for inputs, labels in input_train:
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = train_criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                    scheduler.step()
                correct = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in input_valid:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels).sum().item()
                archive_fitness[i][1] = correct / len(input_valid)
                del model, train_criterion, eval_criterion, optimizer, scheduler, dummy
                torch.cuda.empty_cache()

        if indic == 0:
            for i, temp_in in enumerate(archive_in):
                rounded_in = vectorized_discretize(temp_in, min_, max_)

                model = SearchSpace(rounded_in, num_classes=num_classes, device=device, p=0, stem_multiplier=2, init=False).to(device)
                with torch.no_grad():
                    dummy = torch.randn(1, 3, 32, 32).to(device)
                    _ = model(dummy)

                loader = iter(input_valid)
                inputs, _ = next(loader)
                le = LE.Learnability(model, inputs, device, 2025)
                le.reinitialize(model)
                mean, variance, skewness, kurt, act_num = le.forward()
                mean_input, variance_input, skewness_input, kurt_input = le.calculate_input(inputs)

                mean_score = np.abs(np.array(mean) - mean_input)
                variance_score = np.abs(np.array(variance) - variance_input)
                skewness_score = np.abs(np.array(skewness) - skewness_input)
                kurt_score = np.abs(np.array(kurt) - kurt_input)

                MinMaxScaler(mean_score)
                MinMaxScaler(variance_score)
                MinMaxScaler(skewness_score)
                MinMaxScaler(kurt_score)

                average_list = np.mean(np.vstack([mean_score, variance_score, skewness_score, kurt_score]), axis=0)

                average_list_diff = average_list[:-1]
                average_list_next = average_list[1:]

                LE_score = np.sum(indicator(average_list_diff, average_list_next))
                archive_fitness[i][1] = -LE_score
                if hasattr(le, "clear"):
                    le.clear()
                del le, model, inputs, dummy, mean, variance, skewness, kurt, act_num
                del mean_input, variance_input, skewness_input, kurt_input, mean_score, variance_score, skewness_score, kurt_score, average_list, average_list_diff, average_list_next
                torch.cuda.empty_cache()
    
    in_new = np.concatenate((archive_in, curr_in), axis=0)
    fitness_new = np.concatenate((archive_fitness, curr_fit), axis=0)
    
    pareto_2 = pareto.Pareto_(in_new, fitness_new)
    curr_archiving_in, curr_archiving_fit = pareto_2.pareto() 

    if curr_archiving_in.shape[0] > thresh:
        clear_ = archive.clear_archiving(curr_archiving_in, curr_archiving_fit, mesh_div, min_, max_, particals)
        curr_archiving_in, curr_archiving_fit = clear_.clear_(thresh)

    adj_set = {tuple(subset) for subset in archive_in}
    curr_set = {tuple(subset) for subset in curr_archiving_in}
    difference = len(curr_set.symmetric_difference(adj_set))
    
    return curr_archiving_in, curr_archiving_fit, difference


def update_gbest(archiving_in, archiving_fit, mesh_div, min_, max_, particals):
    get_g = archive.get_gbest(archiving_in, archiving_fit, mesh_div, min_, max_, particals)
    return get_g.get_gbest()
