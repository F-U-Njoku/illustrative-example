"""
Multivariate
Wrapper	NB
Multi	-Z Accuracy, F1
Sequential	->	SFS
"""
from utility import *

df = pd.read_csv("../Telecommunications_Industry/telco_clean.csv")
X = df.drop(columns=['Churn'])
y = df["Churn"]

# Wrapper
nb = GaussianNB()
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
        X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,temp_idx], y, test_size=0.30, random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        nb.fit(X_res, y_res)
        y_pred = nb.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metric = np.average([acc, f1])
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
