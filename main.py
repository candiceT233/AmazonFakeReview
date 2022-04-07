import time

from model_api import *
#import numpy as

filenames = ['amazon_ml_1.csv']
models_api = {
    'knn': knn,
    'svm': svm,
    'ensembleBaggingClassifier': ensembleBaggingClassifier,
    'ensembleAdaBoostClassifier': ensembleAdaBoostClassifier,
    'ensembleVoting': ensembleVoting
}

results = []
for model_name, model_api in models_api.items():
    for filename in filenames:
        try:
            start_time = time.time()
            f = getattr(model_api, model_name + '_model')
            end_time = time.time()
            result = f(filename)
            result['run_time'] = round(end_time - start_time, 4)
            results.append(result)
        except Exception as e:
            continue

import  pandas as pd
df = pd.DataFrame(results)
df.to_csv('model_train_result.csv')
