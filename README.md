# AmazonFakeReview

## TODO:
Add Epinion network code and output

## Preprocess Data

```bash
python3 amazon_preprocess.py
```

## Train Machine Learning Models

```bash
python3 main.py
```

## Observe Datasets as Networks
```bash
python3 amazon_network.py
```

## Input
- datasets
    - Video_Games_5.json
    - soc-sign-epinions.txt
    - english_stopwords.txt
    - amazon_ml_1.csv

## Output
ML Results: model_train_result.txt \
Other Results in outputs file.

## Reference
- Justifying recommendations using distantly-labeled reviews and fined-grained aspects \
Jianmo Ni, Jiacheng Li, Julian McAuley Empirical Methods in Natural Language Processing (EMNLP), 2019
