"""
HMultivariate
Hybrid	mRMR + NB
Many -> Relevance/Redundancy, Accuracy, F1, AUC
Exponential	->	Exhaustive
"""

from utility import *

# Filter
X_res, y_res = smote.fit_resample(X, y)
num_features = round(X.shape[1]*0.80)
start = time()
idx,_,_ = MRMR.mrmr(X_res.values, y_res.values, n_selected_features=num_features)
filter_time = time()-start


# Wrapper
nb = GaussianNB()
subsets = generate_subsets(idx)

best_metric = 0
best_idx = []

start = time()
for subset in subsets:
    X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, list(subset)], y, test_size=0.30, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    nb.fit(X_res, y_res)
    y_pred = nb.predict(X_test)
    # model accuracy
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    metric = np.average([acc, f1, auc])
    if metric > best_metric:
        best_metric = metric
        best_idx = subset
wrapper_time = time()-start

print(f"Time: {filter_time+wrapper_time}")
print(f"Selected features: {X.columns[list(best_idx)]}")
