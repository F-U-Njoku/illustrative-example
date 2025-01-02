"""
Univariate
Filter
Multi	-> Gini, mutual info
Sequential	-> Ranking-based
"""
from utility import *

X, y = smote.fit_resample(X, y)

# Compute Gini index scores and mutual information
start = time()
gini_score = gini_index.gini_index(X.values, y.values)
mi_score = mutual_info_classif(X, y)
fs_time = time() - start

# Rank features by scores
gini_rank = gini_index.feature_ranking(gini_score)
print(gini_rank)
mi_rank = np.argsort(-mi_score)  # Descending rank for mutual information
print(mi_rank)

gini_dict = dict(zip(X.columns[gini_rank], range(1, len(gini_rank) + 1)))
mi_dict = dict(zip(X.columns[mi_rank], range(1, len(mi_rank) + 1)))

# Average ranks
avg_rank = {k: (gini_dict[k] + mi_dict[k]) / 2 for k in X.columns}
avg_rank = dict(sorted(avg_rank.items(), key=lambda item: item[1]))

print(f"Time: {fs_time} seconds")
print(avg_rank.keys())
