'''
Anomaly Detection on Event Logs with a Scarcity of Labels
'''
import numpy as np
from gensim.models import Word2Vec
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

class W2VLOF():
    def __init__(self,window=3, min_count=1):
        self.name= 'W2V-LOF'
        self.window = window
        self.min_count = min_count

    def create_models(self, cases, window, min_count):
        '''
        Creates a word2vec model
        '''
        model = Word2Vec(
            window=window,
            min_count=min_count,
            workers=8)
        model.build_vocab(cases)
        model.train(cases, total_examples=len(cases), epochs=10)

        return model

    def average_feature_vector(self, cases, model):
        '''
        Computes average feature vector for each trace
        '''
        vectors = []
        for case in cases:
            case_vector = []
            for token in case:
                try:
                    case_vector.append(model.wv[token])
                except KeyError:
                    pass
            vectors.append(np.array(case_vector).mean(axis=0))  # In the original author's implementation, the `np.double` type was not used, resulting in insufficient precision for mean calculations. Even when the content is identical, the computed `mean_vector` can differ. This limitation leads to acceptable performance on some datasets, but in reality, it cannot detect anomalies such as ‘early’ or ‘late’ type.

        return vectors

    def fit(self,dataset):
        return self


    def detect(self, dataset):
        cases = []
        for case in dataset.features[0].tolist():
            if 0 in case:
                cases.append(list(map(str, case[:case.index(0)])))
            else:
                cases.append(list(map(str, case)))
        # generate model
        self.word2vecModel = self.create_models(cases, self.window, self.min_count)
        # calculating the average feature vector for each sentence (trace)
        vectors = self.average_feature_vector(cases, self.word2vecModel)
        # normalization
        self.scl = StandardScaler()
        vectors = self.scl.fit_transform(vectors)

        self.model = LocalOutlierFactor(n_jobs=8)

        self.model.fit(vectors)

        scores = - self.model.negative_outlier_factor_  # 越接近1，越好， 越接近正无穷，越差

        if scores.max()==1 and scores.min()==1:
            trace_level_abnormal_scores = scores
        else:
            trace_level_abnormal_scores =  (scores-scores.min())/(scores.max()-scores.min())


        return trace_level_abnormal_scores, None, None
