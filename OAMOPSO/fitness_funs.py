import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from EvoNet import *
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


def fitness_(in_, input_train, input_valid, indic, device, min_, max_, constraint, use_elite_update=True):
    rounded_in = vectorized_discretize(in_, min_, max_)
    temp_in = rounded_in.tolist()
    print('candidate models:', temp_in)

    dataset = getattr(input_train, 'dataset', None)
    base_dataset = getattr(dataset, 'dataset', dataset)
    if base_dataset is not None and hasattr(base_dataset, 'classes'):
        num_classes = len(base_dataset.classes)
    else:
        num_classes = 10

    model = SearchSpace(rounded_in, num_classes=num_classes, device=device, p=0, stem_multiplier=2, init=False).to(
        device)

    fit_1 = sum(v.numel() for v in model.parameters()) / 1e6

    if use_elite_update:
        if indic[-1] == 1:
            print('need to train from scratch')
            train_criterion, eval_criterion, optimizer, scheduler = build_train_Optimizer_Loss(model, 0.9, 0.025, 3e-4,
                                                                                               epochs=8,
                                                                                               device=device)

            for epoch in range(8):
                model.train()
                running_loss = 0.0
                for inputs, labels in input_train:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = train_criterion(outputs, labels)
                    if not torch.isfinite(loss):
                        print('Non-finite loss encountered, skip training for this model.')
                        fit_2 = 0.1
                        del model
                        torch.cuda.empty_cache()
                        print('the fitness of model:', [fit_1, fit_2])
                        if constraint is not None and fit_1 < constraint:
                            fit_1 = 10
                            fit_2 = -0.1
                        return [fit_1, -fit_2]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    running_loss += loss.item()
                print(f"Epoch {epoch + 1}, Loss: {running_loss / len(input_train):.4f}")
                scheduler.step()
            correct = 0
            total = 0
            all_pred = []
            all_label = []
            model.eval()
            with torch.no_grad():
                for inputs, labels in input_valid:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    all_pred.append(predicted.cpu())
                    all_label.append(labels.cpu())
            acc = correct / total if total > 0 else 0.0
            if len(all_pred) > 0:
                all_pred = torch.cat(all_pred)
                all_label = torch.cat(all_label)
                unique_pred, counts_pred = torch.unique(all_pred, return_counts=True)
                unique_label, counts_label = torch.unique(all_label, return_counts=True)
                pred_info = {int(k): int(v) for k, v in zip(unique_pred, counts_pred)}
                label_info = {int(k): int(v) for k, v in zip(unique_label, counts_label)}
                print('pred distribution:', pred_info)
                print('label distribution:', label_info)
            print('Network architecture is:', temp_in)
            print('Validation accuracy: {:.2f}%'.format(acc * 100))
            fit_2 = acc

        elif indic[-1] == 0:
            loader = iter(input_valid)
            inputs, _ = next(loader)
            le = LE.Learnability(model, inputs, device, 2025)
            le.reinitialize(model)
            mean, variance, skewness, kurt, _ = le.forward()
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

            fit_2 = LE_score
            del le, mean, variance, skewness, kurt
            del mean_input, variance_input, skewness_input, kurt_input
            torch.cuda.empty_cache()
    else:
        loader = iter(input_valid)
        inputs, _ = next(loader)
        le = LE.Learnability(model, inputs, device, 2025)
        le.reinitialize(model)
        mean, variance, skewness, kurt, _ = le.forward()
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
        fit_2 = LE_score
        del le, mean, variance, skewness, kurt
        del mean_input, variance_input, skewness_input, kurt_input
        torch.cuda.empty_cache()
    model.cpu()
    del model
    torch.cuda.empty_cache()
    print('the fitness of model:', [fit_1, -fit_2])
    if constraint is not None and fit_1 < constraint:
        fit_1 = 10
        fit_2 = -0.1
    return [fit_1, -fit_2]
