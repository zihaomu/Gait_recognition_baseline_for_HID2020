import torch.utils.data as tordata
import random
from utils.random import random_select


class WeightSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while (True):
            sample_indices = list()
            # print("len batch_size", self.batch_size[0], "len dataset", len(self.dataset.label_set))
            pid_list = random.sample(list(self.dataset.label_set),self.batch_size[0])
            for pid in pid_list:
                _index = self.dataset.index_dict.loc[pid, :].values
                _index = _index[_index > 0].flatten().tolist()
                _index = random_select(_index, k=self.batch_size[1])
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size