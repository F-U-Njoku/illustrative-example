"""
Univariate
Hybrid	-> Gini + ULR
Many	> Gini, Accuracy, Precision, Recall
Sequential -> Ranking-based
"""
from utility import *

acc_score = []
prec_score =  []
rec_score = []

start = time()
for column in X.columns:
    X_train, X_test, y_train, y_test = train_test_split(X[column].values.reshape(-1, 1),
                                                        y, test_size=0.30, random_state=42)

    X_res, y_res = smote.fit_resample(X_train, y_train)

    lr.fit( X_res, y_res)
    y_pred = lr.predict(X_test)

    acc_score.append(accuracy_score(y_test, y_pred))
    prec_score.append(precision_score(y_test, y_pred))
    rec_score.append(recall_score(y_test, y_pred))
fs_time = time()-start

X, y = smote.fit_resample(X, y)

start = time()
# obtain the gini_index score of each feature
score = gini_index.gini_index(X.values, y.values)
fs_time2 = time()-start


# Rank features by scores
gini_rank = gini_index.feature_ranking(score)
acc_rank = np.argsort(-np.array(acc_score))
prec_rank = np.argsort(-np.array(prec_score))
rec_rank = np.argsort(-np.array(rec_score))

gini_dict = dict(zip(X.columns[gini_rank], range(1, len(gini_rank) + 1)))
acc_dict = dict(zip(X.columns[acc_rank], range(1, len(acc_rank) + 1)))
prec_dict = dict(zip(X.columns[prec_rank], range(1, len(prec_rank) + 1)))
rec_dict = dict(zip(X.columns[rec_rank], range(1, len(rec_rank) + 1)))

# Average ranks
avg_rank = {k: np.average([gini_dict[k], acc_dict[k], prec_dict[k], rec_dict[k]])  for k in X.columns}
avg_rank = dict(sorted(avg_rank.items(), key=lambda item: item[1]))

print(f"Time: {fs_time+fs_time2}")
print(avg_rank.keys())
