"""
Multivariate
Filter	-> mRMR
Single	-> Relevance/Redundancy
Sequential	-> SFS
"""
from utility import *
X, y = smote.fit_resample(X, y)
# Wrapper
subsets = list(range(0,X.shape[1]))

best_metric = 0
best_idx = []
flag = True

start = time()
while flag:
    selected_feature = ""
    for feature in subsets:

        temp_idx = best_idx.copy()
        temp_idx.append(feature)

        X_train = X.iloc[:,temp_idx]
        # mrmr
        _, mrmr_score, _ = MRMR.mrmr(X_train.values, y.values, n_selected_features=len(temp_idx))
        # mrmr_score = calculate_mrmr(X_train, y)

        metric = np.average(mrmr_score)
        if metric > best_metric:
            best_metric = metric
            selected_feature = feature
    if selected_feature != "":
        subsets.remove(selected_feature)
        best_idx.append(selected_feature)
    else:
        flag = False
wrapper_time = time()-start

print(f"Time: {wrapper_time}")
print(f"Selected features: {X.columns[list(best_idx)]}")
