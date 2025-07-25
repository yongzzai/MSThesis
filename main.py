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
    print(len(dataset.features))

    print(dataset.features[0].shape)
    print(dataset.features[1].shape)
    print(dataset.features[2].shape)

    # res = [fit_and_eva(d, **ad) for ad in ads for d in dataset_names]