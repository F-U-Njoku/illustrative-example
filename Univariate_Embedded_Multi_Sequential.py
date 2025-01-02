"""
Univariate
Embedded -> DT
Multi	-> Split criterion, Accuracy
Sequential	->	Ranking-based
"""
from utility import *

dt = DecisionTreeClassifier(criterion="entropy", random_state=0)
scores = []

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

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)
wrapper_time = time()-start

X, y = smote.fit_resample(X, y)

start = time()
dt.fit(X, y)
wrapper_time2 = time()-start

var = dt.feature_importances_

# Rank features by scores
coef_rank = np.argsort(-var)
acc_rank = np.argsort(-np.array(scores))

coef_dict = dict(zip(X.columns[coef_rank], range(1, len(coef_rank) + 1)))
acc_dict = dict(zip(X.columns[acc_rank], range(1, len(acc_rank) + 1)))


# Average ranks
avg_rank = {k: np.average([coef_dict[k], acc_dict[k]]) for k in X.columns}
avg_rank = dict(sorted(avg_rank.items(), key=lambda item: item[1]))

print(f"Time: {wrapper_time+wrapper_time2} seconds")
print(avg_rank.keys())
