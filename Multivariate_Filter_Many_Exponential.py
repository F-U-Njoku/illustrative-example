from utility import *
"""
Wrapper -> NB
The methods for evaluation are multivariate
Exponential is used for the search --> Exhaustive
Many (4) objective --> mRMR, CMIM, JMI, MIM
"""

# Filter
subsets = generate_subsets(range(0, X.shape[1]))

best_metric = 0
best_idx = []

start = time()
for subset in subsets:
    X_train = X.iloc[:, list(subset)]

    # mrmr
    #_, mrmr_score, _ = MRMR.mrmr(X_train.values, y.values, n_selected_features=len(list(subset)))
    mrmr_score = calculate_mrmr(X_train, y)
    # cmim
    #_, cmim_score, _ = CMIM.cmim(X_train.values, y.values, n_selected_features=len(list(subset)))
    cmim_score = calculate_cmim(X_train, y)
    # JMI
    #_, jmi_score, _ = JMI.jmi(X_train.values, y.values, n_selected_features=len(list(subset)))
    jmi_score = calculate_jmi(X_train, y)
    # mim
    #_, mim_score, _ = MIFS.mifs(X_train.values, y.values, n_selected_features=len(list(subset)))
    mim_score = calculate_fcbf(X_train, y)

    metric = np.average([np.average(mrmr_score), np.average(cmim_score),
                         np.average(jmi_score), np.average(mim_score)])
    if metric > best_metric:
        best_metric = metric
        best_idx = subset
wrapper_time = time()-start

print(f"Time: {wrapper_time}")
print(f"Best index: {list(best_idx)}")
print(f"Selected features: {X.columns[list(best_idx)]}")
