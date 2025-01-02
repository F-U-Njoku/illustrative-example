"""
Univariate
Embedded -> DT
Many -> Split criterion, Accuracy, F1, AUC
Sequential	-> Ranking-based
"""
from utility import *

dt = DecisionTreeClassifier(criterion="entropy", random_state=0)
acc_scores = []
f1_scores = []
auc_scores = []

# For each column, train a separate decision tree
start = time()
for column in X.columns:
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X[column].values.reshape(-1, 1),
                                                        y, test_size=0.30, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Fit and predict
    dt.fit(X_res, y_res)
    y_pred = dt.predict(X_test)

    # Calculate and print accuracy
    acc_scores.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    auc_scores.append(roc_auc_score(y_test, y_pred))
wrapper_time = time() - start

X, y = smote.fit_resample(X, y)

start = time()
dt.fit(X, y)
wrapper_time2 = time() - start

var = dt.feature_importances_


# Rank features by scores
coef_rank = np.argsort(-var)
acc_rank = np.argsort(-np.array(acc_scores))
f1_rank = np.argsort(-np.array(f1_scores))
auc_rank = np.argsort(-np.array(auc_scores))

coef_dict = dict(zip(X.columns[coef_rank], range(1, len(coef_rank) + 1)))
acc_dict = dict(zip(X.columns[acc_rank], range(1, len(acc_rank) + 1)))
f1_dict = dict(zip(X.columns[f1_rank], range(1, len(f1_rank) + 1)))
auc_dict = dict(zip(X.columns[auc_rank], range(1, len(auc_rank) + 1)))

# Average ranks+
avg_rank = {k: np.average([coef_dict[k], acc_dict[k], f1_dict[k], auc_dict[k]]) for k in X.columns}
avg_rank = dict(sorted(avg_rank.items(), key=lambda item: item[1]))

print(f"Time: {wrapper_time+wrapper_time2} seconds")
print(avg_rank.keys())
