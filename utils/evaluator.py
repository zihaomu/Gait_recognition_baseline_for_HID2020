import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

def cuda_dist(x, y):  # probe_seq_x, gallery_seq_y
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda().squeeze(1)
    a = torch.sum(x ** 2, 1).unsqueeze(1)
    b = torch.sum(y ** 2, 1).unsqueeze(1).transpose(0, 1)
    c = 2 * torch.matmul(x, y.transpose(0, 1))
    dist = a + b - c
    dist = torch.sqrt(F.relu(dist))
    return dist

class Evaluator(object):

    def __init__(self, gallery_data, probe_data, config):

        self.config = config
        self.gallery_feature,  self.gallery_vID, self.gallery_label = gallery_data[0], gallery_data[1], gallery_data[2]
        self.probe_feature, self.probe_vID = probe_data[0], probe_data[1]
        
        self.gallery_feature = np.array(self.gallery_feature)
        self.probe_feature = np.array(self.probe_feature)

        self.idx = list()
        # print("gallery len", len(self.gallery_label))
        self.predict_lable = None

    def save_submission(self, data):
        csv_path = os.path.join(self.config.train.dir, "submission.csv")
        data.to_csv(csv_path, index=False)
        print("CSV file saved successfully!!, csv path is :", csv_path)
        print("Note that the submitted submission.csv file must be compressed into a zip archive.")

    def load_submission(self):
        submission_path = self.config.test.SampleSubmission_dir
        if not os.path.exists(submission_path):
            print("")
            exit()
        pd_data = pd.read_csv(submission_path)

        return pd_data

    def get_dist(self, dist):
        self.idx = dist

    def run(self):
        # pred_label = np.asarray([[gallery_label[idx[i][j]] for j in range(num_rank) ] for i in range(len(idx))])
        dist = cuda_dist(self.probe_feature, self.gallery_feature)
        self.idx = dist.sort(1)[1].cpu().numpy()
        error_count = 0
        pd_data = self.load_submission()
        for i in range(len(self.probe_vID)):
            temp_label = self.gallery_label[self.idx[i][0]]
            vID = self.probe_vID[i]
            index = pd_data[pd_data["videoID"] == vID].index.tolist()
            if len(index) == 1:
                pd_data.iloc[index[0], 1] = temp_label
            else:
                error_count = error_count + 1
                print("vID = ", vID, "index", index)
        print("count = ", error_count)
        self.save_submission(pd_data)





