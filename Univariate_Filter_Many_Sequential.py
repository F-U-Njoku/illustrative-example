"""
Univariate	
Filter	
Many -> Gini, mutual info, Chi square, SU
Sequential	-> Ranking-based
"""
from utility import *

X, y = smote.fit_resample(X, y)

# Compute Gini index scores and mutual information
start = time()
gini_score = gini_index.gini_index(X.values, y.values)
mi_score = mutual_info_classif(X, y)
chi2_score, _ = chi2(X, y)
su_score = [mi.su_calculation(X[x_].values, y.values) for x_ in X.columns]
fs_time = time() - start


# Rank features by scores
gini_rank = gini_index.feature_ranking(gini_score)
mi_rank = np.argsort(-mi_score)  # Descending rank for mutual information
chi_rank = np.argsort(-chi2_score)
su_rank = np.argsort(-np.array(su_score))  # Descending rank for symmetric uncertainty



gini_dict = dict(zip(X.columns[gini_rank], range(1, len(gini_rank) + 1)))
mi_dict = dict(zip(X.columns[mi_rank], range(1, len(mi_rank) + 1)))
chi_dict = dict(zip(X.columns[chi_rank], range(1, len(chi_rank) + 1)))
su_dict = dict(zip(X.columns[su_rank], range(1, len(su_rank) + 1)))

# Average ranks
avg_rank = {k: (gini_dict[k] + mi_dict[k] + chi_dict[k] + su_dict[k]) / 4 for k in X.columns}
avg_rank = dict(sorted(avg_rank.items(), key=lambda item: item[1]))

print(f"Time: {fs_time} seconds")
print(avg_rank.keys())
