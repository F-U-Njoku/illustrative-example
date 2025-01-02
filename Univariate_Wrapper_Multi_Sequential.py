"""
Univariate
Wrapper	-> Univariate Logistic Regression
Multi	-> Accuracy, F1
Sequential -> Ranking-based
"""
from utility import *

scores = []

start = time()
for column in X.columns:
    X_train, X_test, y_train, y_test = train_test_split(X[column].values.reshape(-1, 1),
                                                        y, test_size=0.30, random_state=42)

    X_res, y_res = smote.fit_resample(X_train, y_train)

    lr.fit( X_res, y_res)
    y_pred = lr.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metric = np.average([acc, f1])
    scores.append(metric)
wrapper_time = time()-start

var_dict = dict(zip(X.columns, scores))

sorted_feats = [key for key, value in sorted(var_dict.items(), key=lambda item: item[1], reverse=True)]
sorted_values = [value for key, value in sorted(var_dict.items(), key=lambda item: item[1], reverse=True)]

print(f"Time: {wrapper_time}")
print(sorted_feats)
print(sorted_values)
