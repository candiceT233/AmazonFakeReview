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
python3 epinion_network.py
```

## Input
- datasets (put in this folder)
    - Video_Games_5.json (https://nijianmo.github.io/amazon/index.html#subsets)
    - soc-sign-epinions.txt (https://snap.stanford.edu/data/soc-sign-epinions.html)
    - english_stopwords.txt (https://pythonspot.com/nltk-stop-words/)
    - amazon_ml_1.csv (generated from amazon_preprocess.py)

## Output
ML Results: model_train_result.txt \
Other Results are in the outputs file.

## Reference
- Justifying recommendations using distantly-labeled reviews and fined-grained aspects \
Jianmo Ni, Jiacheng Li, Julian McAuley Empirical Methods in Natural Language Processing (EMNLP), 2019
