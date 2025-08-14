import os
from utils.dataset import Dataset

from utils.eval import cal_best_PRF
from utils.fs import EVENTLOG_DIR, ROOT_DIR
import argparse
import time


if __name__ == '__main__':

    dataset_names = os.listdir(EVENTLOG_DIR)
    dataset_names.sort()
    if 'cache' in dataset_names:
        dataset_names.remove('cache')

    print('number of datasets:' + str(len(dataset_names)))

    d = 'BPIC20_International'
    dataset_names = [name for name in dataset_names if d in name]
    
    start_time = time.time()
    dataset = Dataset(dataset_names[0])
    
    print(dataset.attribute_dims)

    from model.model import GAIN

    gain = GAIN(hidden_dim=64, num_enc_layers=2, num_dec_layers=2,
                 enc_dropout=0.3, dec_dropout=0.3, batch_size=64, epochs=15, lr=0.0004,
                 seed=42)

    gain.fit(dataset)

    end_time = time.time()
    duration = (end_time - start_time).__round__(3)

    trace_level_anomaly_scores, event_level_anomaly_scores, attr_level_anomaly_scores = gain.detect(dataset)      

    event_target = dataset.binary_targets.sum(2).flatten()
    event_target[event_target > 1] = 1

    precision_t, recall_t, f1_t, aupr_t = cal_best_PRF(dataset.case_target, trace_level_anomaly_scores)
    precision_e, recall_e, f1_e, aupr_e = cal_best_PRF(event_target, event_level_anomaly_scores.flatten())
    precision_a, recall_a, f1_a, aupr_a = cal_best_PRF(dataset.binary_targets.flatten(), attr_level_anomaly_scores.flatten())

    print("Trace-level")
    print(f"precision: {precision_t}, recall: {recall_t}, f1: {f1_t}, aupr: {aupr_t}")
    print("Event-level")
    print(f"precision: {precision_e}, recall: {recall_e}, f1: {f1_e}, aupr: {aupr_e}")
    print("Attribute-level")
    print(f"precision: {precision_a}, recall: {recall_a}, f1: {f1_a}, aupr: {aupr_a}")
    print(f"Time Consumption: {duration}")
