import os
import numpy as np
from utils.dataset import Dataset
from scipy.ndimage import uniform_filter1d

from utils.eval import cal_best_PRF
from utils.fs import EVENTLOG_DIR, ROOT_DIR
import argparse


if __name__ == '__main__':

    dataset_names = os.listdir(EVENTLOG_DIR)
    dataset_names.sort()
    if 'cache' in dataset_names:
        dataset_names.remove('cache')

    dataset_names_syn = [name for name in dataset_names if (
                                                        'gigantic' in name
                                                        or 'huge' in name
                                                        or 'large' in name
                                                        or 'medium' in name
                                                        or 'p2p' in name
                                                        or 'paper' in name
                                                        or 'small' in name
                                                        or 'wide' in name
    )]

    dataset_names_real = list(set(dataset_names)-set(dataset_names_syn))
    dataset_names_real.sort()

    print('number of datasets:' + str(len(dataset_names)))
    
    print(dataset_names[-9])
    dataset = Dataset(dataset_names[-9])

    from model.model import GAIN

    gain = GAIN(hidden_dim=64, num_enc_layers=2, num_dec_layers=2,
                enc_dropout=0.2, dec_dropout=0.3, batch_size=64, epochs=18, lr=0.0004, seed=42)
    gain.fit(dataset)

    res = gain.detect(dataset)      # Shape(num_cases, seq_len, num_attr)

    attr_level_anomaly_scores = []
    attr_level_anomaly_labels = []

    for case_idx in range(dataset.num_cases):
        current_res = 1 - res[case_idx,:,:]                     # Shape(seq_len, num_attr)
        current_label = dataset.binary_targets[case_idx,:,:]    # Shape(seq_len, num_attr)

        value_mask = dataset.features[0][case_idx] != 0 # Shape(seq_len,)
        value_mask[0] = False

        current_res = current_res[value_mask]
        current_label = current_label[value_mask]

        attr_level_anomaly_scores.append(current_res)
        attr_level_anomaly_labels.append(current_label)

    attr_level_anomaly_scores = np.concatenate(attr_level_anomaly_scores, axis=0).flatten()
    attr_level_anomaly_labels = np.concatenate(attr_level_anomaly_labels, axis=0).flatten()

    precisions, recalls, f1s, aupr = cal_best_PRF(attr_level_anomaly_labels, attr_level_anomaly_scores)

    print("Precision", precisions)
    print("Recall", recalls)
    print("F1 Score", f1s)
    print("AUPR", aupr)
