import random

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import skew, kurtosis


def compute_feature_statistics(feature_map):
    feature_map_np = feature_map.cpu().numpy()
    mean = np.mean(feature_map_np)
    variance = np.var(feature_map_np)
    skewness = skew(feature_map_np.flatten())
    kurt = kurtosis(feature_map_np.flatten())

    return mean, variance, skewness, kurt


def calculate_input(inputs):
    mean, variance, skewness, kurt = compute_feature_statistics(inputs)
    return mean, variance, skewness, kurt


class Learnability(object):
    def __init__(self, model=None, inputs=None, device='cuda', seed=random.randint(1, 1000)):
        self.model = model
        self.inputs = inputs
        self.device = device
        self.seed = seed
        self.interFeature = []

        self.FeatureMean = []
        self.FeatureStd = []
        self.FeatureS = []
        self.FeatureK = []
        self.act = None
        self.hooks = []

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

        self.FeatureMean.clear()
        self.FeatureStd.clear()
        self.FeatureS.clear()
        self.FeatureK.clear()

        torch.cuda.empty_cache()

    def reinitialize(self, model=None, seed=None):
        if model is not None:
            self.model = model
            self.register_hook(self.model)

    def forward(self):

        with torch.no_grad():
            self.model.to(self.device)
            self.model.forward(self.inputs.to(self.device))

            for feature in self.interFeature:
                mean, variance, skewness, kurt = compute_feature_statistics(feature)
                self.FeatureMean.append(mean)
                self.FeatureStd.append(variance)
                self.FeatureS.append(skewness)
                self.FeatureK.append(kurt)
            self.interFeature.clear()
        return self.FeatureMean, self.FeatureStd, self.FeatureS, self.FeatureK, self.act

    def register_hook(self, model):
        self.act = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                h = module.register_forward_hook(self.hook_in_forward)
                self.act += 1
                self.hooks.append(h)

    def hook_in_forward(self, module, input, output):
        # if isinstance(input, tuple) and len(input[0].size()) == 4:
        #     self.interFeature.append(output.detach())
        self.interFeature.append(output.detach())

