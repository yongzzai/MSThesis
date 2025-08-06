import os
import pandas as pd

from baseline.GAE.gae import GAE
from baseline.GAMA.gama import GAMA
from baseline.GRASPED.grasped import GRASPED
from baseline.LAE.lae import LAE
from baseline.Sylvio import W2VLOF
from baseline.VAE.vae import VAE
from baseline.VAEOCSVM.vaeOCSVM import VAEOCSVM
from baseline.bezerra import SamplingAnomalyDetector, NaiveAnomalyDetector

# Tensorflow 사용
# from baseline.dae import DAE
# from baseline.binet.binet import BINetv3, BINetv2

from baseline.boehmer import LikelihoodPlusAnomalyDetector
from baseline.leverage import Leverage
from utils.dataset import Dataset

from utils.eval import cal_best_PRF
from utils.fs import EVENTLOG_DIR, ROOT_DIR


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
    
    print(dataset_names[0])
    dataset = Dataset(dataset_names[0])

    print('====')
    print(dataset.DataChunks[0].act_pos)
    print(dataset.DataChunks[0].act_origin)
    print("----")
    from torch_geometric.loader import DataLoader
    import torch

    loader = DataLoader(
        dataset=dataset.DataChunks,
        batch_size=5,
        shuffle=True,
        follow_batch=['x', 'seq']
    )

    first_batch = next(iter(loader))
    print("x shape:", first_batch.x.shape)
    print("seq shape:", first_batch.seq.shape)
    print("batch_idx unique:",  torch.unique(first_batch.x_batch))
    print("seq batch shape:", first_batch.seq_batch)
    print("act_pos shape:", first_batch.act_pos.shape)
    print("act_origin shape:", first_batch.act_origin.shape)

    from model.model import GAIN

    gain = GAIN(hidden_dim=64, num_gru_layer=2, batch_size=5, epochs=10, lr=0.001, seed=42)

    gain.fit(dataset)