import time

from model_api import *
#import numpy as

filenames = ['amazon_ml_1.csv']
models_api = {
    'knn': knn,
    'svm': svm,
    'ensembleVoting': ensembleVoting,
    'ensembleBaggingClassifier': ensembleBaggingClassifier,
    'ensembleAdaBoostClassifier': ensembleAdaBoostClassifier
}

results = []
for model_name, model_api in models_api.items():
    for filename in filenames:
        try:
            start_time = time.time()
            f = getattr(model_api, model_name + '_model')
            result = f(filename)
            end_time = time.time()
            result['run_time_sec'] = round(end_time - start_time, 3)
            print('results:', result)
            #print(f"Run Time (sec): {result['run_time_sec']}")
            results.append(result)
        except Exception as e:
            continue

import  pandas as pd
df = pd.DataFrame.from_dict(results)

stamp= str(round(time.time()))[-4:]

df.to_csv(f'outputs/{stamp}_model_train_result.csv')
