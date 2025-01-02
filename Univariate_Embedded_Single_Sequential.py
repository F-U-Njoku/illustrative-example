"""
Univariate
Embedded	-> DT
Single	-> plit criterion
Sequential	->	Ranking-based
"""
from utility import *

X, y = smote.fit_resample(X, y)
dt = DecisionTreeClassifier(criterion="entropy", random_state=0)

start = time()
dt.fit(X, y)
wrapper_time = time()-start


var = dt.feature_importances_

var_dict = dict(zip(X.columns, var))

sorted_feats = [key for key, value in sorted(var_dict.items(), key=lambda item: item[1], reverse=True)]

print(f"Time: {wrapper_time}")
print(sorted_feats)
print(sorted(var, reverse=True))
