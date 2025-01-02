"""
Multivariate
Hybrid	-> mRMR + NB
Multi	-> Relevance/Redundancy, Accuracy
Sequential	-> SFS
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
subsets = idx.tolist()

best_metric = 0
best_idx = []
flag = True

start = time()
while flag:
    selected_feature = ""

    for feature in subsets:

        temp_idx = best_idx.copy()
        temp_idx.append(feature)

        X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,temp_idx], y, test_size=0.30, random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        nb.fit(X_res, y_res)
        y_pred = nb.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        if acc > best_metric:
            best_metric = acc
            selected_feature = feature

    if selected_feature != "":
        subsets.remove(selected_feature)
        best_idx.append(selected_feature)
    else:
        flag = False
wrapper_time = time()-start

print(f"Time: {filter_time+wrapper_time}")
print(f"Selected features: {X.columns[list(best_idx)]}")
