
from utils.dataset import Dataset
from utils.eval import cal_best_PRF
from utils.fs import EVENTLOG_DIR, ROOT_DIR
from utils.util import get_model_args

import os
import time
import warnings
import argparse
import pandas as pd

from model.model import DHiM


parser = argparse.ArgumentParser()
parser.add_argument('--embed_dim', '-emb', type=int, default=16)
parser.add_argument('--hidden_dim', '-hid',type=int, default=64)
parser.add_argument('--num_enc_layers', '-el', type=int, default=4)
parser.add_argument('--num_dec_layers', '-dl', type=int, default=2)
parser.add_argument('--enc_dropout', '-ed', type=float, default=0.2)
parser.add_argument('--dec_dropout', '-dd', type=float, default=0.3)
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--epochs', '-epoch', type=int, default=18)
parser.add_argument('--lr', '-lr', type=float, default=0.0002)
parser.add_argument('--seed', '-seed', type=int, default=42)
args = parser.parse_args()
model_args = get_model_args(args)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    dataset_names = os.listdir(EVENTLOG_DIR)
    dataset_names.sort()
    if 'cache' in dataset_names:
        dataset_names.remove('cache')
    
    dataset_names = [name for name in dataset_names if
                     'real-life' not in name
                     and 'synthetic' not in name]
    
    print('number of datasets:' + str(len(dataset_names)))
    
    if args.batch_size < 16:
        dataset_names = [n for n in dataset_names if 'BPIC20' not in n]

    pid = os.getpid()
    date = time.strftime("%m-%d", time.localtime())

    results = pd.DataFrame(columns=['dataset','param_b', 'precision_t', 'recall_t', 'f1_t', 'aupr_t',
                                     'precision_e', 'recall_e', 'f1_e', 'aupr_e',
                                     'precision_a', 'recall_a', 'f1_a', 'aupr_a',
                                     'duration'])
    
    os.makedirs(os.path.join(ROOT_DIR, 'results'), exist_ok=True)
    results_path = os.path.join(ROOT_DIR, 'results', f'result_{date}_{pid}.csv')

    for d in dataset_names:
        try:
            start_time = time.time()
            dataset = Dataset(d)
            print(f"Dataset: {d}")

            dhim = DHiM(**model_args)

            dhim.fit(dataset)

            end_time = time.time()
            duration = (end_time - start_time).__round__(3)

            trace_level_anomaly_scores, event_level_anomaly_scores, attr_level_anomaly_scores = dhim.detect(dataset)

            event_target = dataset.binary_targets.sum(2).flatten()
            event_target[event_target > 1] = 1

            precision_t, recall_t, f1_t, aupr_t = cal_best_PRF(dataset.case_target, trace_level_anomaly_scores)
            precision_e, recall_e, f1_e, aupr_e = cal_best_PRF(event_target, event_level_anomaly_scores.flatten())
            precision_a, recall_a, f1_a, aupr_a = cal_best_PRF(dataset.binary_targets.flatten(), attr_level_anomaly_scores.flatten())

            new_row = {'dataset': d, 'param_b': model_args['batch_size'], 'precision_t': precision_t, 'recall_t': recall_t, 'f1_t': f1_t, 'aupr_t': aupr_t,
                       'precision_e': precision_e, 'recall_e': recall_e, 'f1_e': f1_e, 'aupr_e': aupr_e,
                       'precision_a': precision_a, 'recall_a': recall_a, 'f1_a': f1_a, 'aupr_a': aupr_a,
                       'duration': duration}
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
            results.to_csv(results_path, index=False)

        except Exception as e:
            print(f"Error occurred while processing dataset {d}: {e}")
