"""
Multivariate
Wrapper -> NB
Many -> Accuracy, Precision, Recall, AUC
Exponential ->	Exhaustive
"""
from utility import *

# Wrapper
nb = GaussianNB()
subsets = generate_subsets(range(0, X.shape[1]))

best_metric = 0
best_idx = []

start = time()
for subset in subsets:
    X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,list(subset)], y, test_size=0.30, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    nb.fit(X_res, y_res)
    y_pred = nb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    metric = np.average([acc, precision, recall, auc])
    if metric > best_metric:
        best_metric = metric
        best_idx = subset
wrapper_time = time()-start

print(f"Time: {wrapper_time}")
print(f"Selected features: {X.columns[list(best_idx)]}")
