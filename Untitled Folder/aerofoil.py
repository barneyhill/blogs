import numpy as np         
import pandas as pd
import matplotlib.pyplot as plt
from equadratures import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
data = pd.read_table('airfoil_self_noise.dat',names=["frequency","aoa","chord","vinf","delta","noise"])
data = data[0:200]
features = ['frequency','aoa','chord','vinf','delta']
target   = 'noise'
Xorig = data[features]
y = data[target]
def r2_score(y_true, y_pred):
    print(y_true.shape)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    adj_r2 = (1 - (1 - r2) * ((y_true.shape[0] - 1) /
                              (y_true.shape[0] - y_true.shape[1] - 1)))
    return adj_r2
nu = 1.568e-5
X = Xorig.copy()
X['Re'] = X['chord']*X['vinf']/nu
X=X.drop(columns=['chord','vinf'])
#X['aoa'] = np.abs(X['aoa'])
features = X.keys()
global X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler().fit(X_train)
#scaler = MinMaxScaler((-1,1)).fit(X_train)
X_train_temp = scaler.transform(X_train)
X_test_temp  = scaler.transform(X_test)

y_train = y_train.to_numpy().reshape((y_train.shape[0],1))
y_test = y_test.to_numpy().reshape((y_test.shape[0],1))

model = polytree.PolyTree(max_depth=3,min_samples_leaf=30)
model.get_graphviz(feature_names=['frequency','aoa','chord','vinf','delta'])
model.fit(X_train_temp, y_train)

y_pred_train = model.predict(X_train_temp)
y_pred_test = model.predict(X_test_temp)

r2_train = r2_score(y_train,y_pred_train)
r2_test = r2_score(y_test,y_pred_test)

