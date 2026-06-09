from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import metrics
import numpy as np
import pickle

# ============================================================
# !! TODO - 팀 번호를 수정하세요 !!
# ============================================================
TEAM = 22

# 데이터 설정
categories = ['AnnStat', 'Biometrika', 'JASA', 'JMLR']

train_data = load_files(container_path='text/train', categories=categories,
                        shuffle=True, encoding='utf-8', decode_error='replace')

# ============================================================
# !! TODO - 2-1-1. Naive Bayes Classifier Pipeline을 구성하세요 !!
# 필요한 import를 파일 상단에 추가하세요.
# 예: from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.naive_bayes import MultinomialNB
# Pipeline([('tfidf', TfidfVectorizer(...)), ('clf', MultinomialNB(...))])
# ============================================================
clf_nb = Pipeline([
    ('features', FeatureUnion([
        ('word', TfidfVectorizer(
            strip_accents='unicode',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )),
        ('char', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True
        ))
    ])),
    ('clf', MultinomialNB(alpha=0.05))
])
clf_nb.fit(train_data.data, train_data.target)

# ============================================================
# !! TODO - 2-1-2. SVM Classifier Pipeline을 구성하세요 !!
# 필요한 import를 파일 상단에 추가하세요.
# 예: from sklearn.svm import LinearSVC
# Pipeline([('tfidf', TfidfVectorizer(...)), ('clf', LinearSVC(...))])
# ============================================================
clf_svm = Pipeline([
    ('features', FeatureUnion([
        ('word', TfidfVectorizer(
            strip_accents='unicode',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ('char', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True
        ))
    ])),
    ('clf', LinearSVC(C=0.3, random_state=22, max_iter=10000))
])
clf_svm.fit(train_data.data, train_data.target)

# ╔════════════════════════════════════════════════════════════╗
# ║  DO NOT MODIFY BELOW - 평가 및 모델 저장                  ║
# ╚════════════════════════════════════════════════════════════╝
test_data = load_files(container_path='text/test', categories=categories,
                       shuffle=True, encoding='utf-8', decode_error='replace')
docs_test = test_data.data

# Evaluate Naive Bayes
predicted = clf_nb.predict(docs_test)
print("NB accuracy : %d / %d" % (np.sum(predicted==test_data.target), len(test_data.target)))
print(metrics.classification_report(test_data.target, predicted, target_names=test_data.target_names))
print(metrics.confusion_matrix(test_data.target, predicted))

# Evaluate SVM
predicted = clf_svm.predict(docs_test)
print("\nSVM accuracy : %d / %d" % (np.sum(predicted==test_data.target), len(test_data.target)))
print(metrics.classification_report(test_data.target, predicted, target_names=test_data.target_names))
print(metrics.confusion_matrix(test_data.target, predicted))

# Save models
with open('DMA_project2_team%02d_nb.pkl' % TEAM, 'wb') as f1:
    pickle.dump(clf_nb, f1)

with open('DMA_project2_team%02d_svm.pkl' % TEAM, 'wb') as f2:
    pickle.dump(clf_svm, f2)
