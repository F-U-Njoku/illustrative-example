"""
Multivariate
Filter ->	mRMR
Single	-> Relevance/Redundancy
Exponential	-> Exhaustive
"""
from utility import *

subsets = generate_subsets(range(0, X.shape[1]))
X, y = smote.fit_resample(X, y)

best_metric = 0
best_idx = []

start = time()
for subset in subsets:
    X_train = X.iloc[:, list(subset)]

    # mrmr
    #_, mrmr_score, _ = MRMR.mrmr(X_train.values, y.values, n_selected_features=len(list(subset)))

    mrmr_score = calculate_mrmr(X_train, y)

    metric = np.average(mrmr_score)
    print(metric)

    if metric > best_metric:
        best_metric = metric
        best_idx = subset
wrapper_time = time()-start

print(f"Time: {wrapper_time}")
print(f"Selected features: {X.columns[list(best_idx)]}")
