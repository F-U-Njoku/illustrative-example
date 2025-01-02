"""
Univariate
Filter	-> Gini
Single	-> Gini score
Sequential > Ranking-based
"""
from utility import *

X, y = smote.fit_resample(X, y)

start = time()
# obtain the gini_index score of each feature
score = gini_index.gini_index(X.values, y.values)
# rank features in descending order according to score
fs_time = time()-start

idx = gini_index.feature_ranking(score)

print(f"Time: {fs_time}")
print(X.columns[idx])
print(score)
print(np.flip(np.sort(score)))
