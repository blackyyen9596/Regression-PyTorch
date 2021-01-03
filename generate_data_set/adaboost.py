from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import sklearn.ensemble as se
import sklearn.tree as st
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

from preprocessing import preprocessing as pre
from to_csv import to_csv

train_x, val_x, test_x, train_y, val_y = pre()

# 標準化
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
val_x = ss.transform(val_x)
test_x = ss.transform(test_x)

kf = KFold(n_splits=10, shuffle=False)
fold = 0
train_array = []
val_array = []
val_dic = {}
test_dic = {}
val_dic['id'] = pd.read_csv(r'../ntut-ml-2020-regression/valid-v3.csv').id
test_dic['id'] = pd.read_csv(r'../ntut-ml-2020-regression/test-v3.csv').id

for train_index, test_index in kf.split(train_x):
    # print('train_index:%s , test_index: %s ' % (train_index, test_index))
    trainfold_x = train_x[train_index]
    trainfold_y = train_y[train_index]
    fold += 1
    # 設定AdaBoost參數
    model = se.AdaBoostRegressor(st.DecisionTreeRegressor(criterion="mse",
                                                          splitter='best',
                                                          max_depth=None,
                                                          min_samples_split=2,
                                                          min_samples_leaf=2,
                                                          max_features='auto'),
                                 n_estimators=350,
                                 learning_rate=1.3,
                                 random_state=4)
    # 訓練模型
    model.fit(trainfold_x, trainfold_y)
    # 生成新訓練集
    train_y_pred = model.predict(train_x[test_index])
    for i in range(len(train_y_pred)):
        train_array.append(int(train_y_pred[i]))
    # 生成新驗證集
    val_y_pred = model.predict(val_x)
    val_dic['price' + str(fold)] = val_y_pred
    # 生成新測試集
    teat_y_pred = model.predict(test_x)
    test_dic['price' + str(fold)] = teat_y_pred

# 生成新的訓練集
train_id = pd.read_csv(r'../ntut-ml-2020-regression/train-v3.csv').id
my_submission = pd.DataFrame({'id': train_id, 'adaboost': train_array[:]})
my_submission.to_csv('{}.csv'.format('../result/adaboost/train_adaboost'),
                     index=False)

# 生成新的驗證集
total_price = 0
for i in range(fold):
    total_price += val_dic['price' + str(i + 1)]
avg_price = total_price // fold
val_dic['adaboost'] = avg_price
my_submission = pd.DataFrame(val_dic)
my_submission = my_submission.loc[:, ['id', 'adaboost']]
my_submission.to_csv('{}.csv'.format('../result/adaboost/val_adaboost'),
                     index=False)

print('train_mean_absolute_error',
      int(mean_absolute_error(val_y, my_submission.loc[:, ['adaboost']])))

# 生成新的測試集
total_price = 0
for i in range(fold):
    total_price += test_dic['price' + str(i + 1)]
avg_price = total_price // fold
test_dic['adaboost'] = avg_price
my_submission = pd.DataFrame(test_dic)
my_submission = my_submission.loc[:, ['id', 'adaboost']]
my_submission.to_csv('{}.csv'.format('../result/adaboost/test_adaboost'),
                     index=False)