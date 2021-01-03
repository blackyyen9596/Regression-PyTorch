from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

from preprocessing import preprocessing as pre
from to_csv import to_csv

# 讀取預處理資料
train_x, val_x, test_x, train_y, val_y = pre()

# 標準化
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
val_x = ss.transform(val_x)
test_x = ss.transform(test_x)

kf = KFold(n_splits=10, shuffle=False)
fold = 0
train_array = []
val_dic = {}
test_dic = {}
val_dic['id'] = pd.read_csv(r'../ntut-ml-2020-regression/valid-v3.csv').id
test_dic['id'] = pd.read_csv(r'../ntut-ml-2020-regression/test-v3.csv').id

for train_index, test_index in kf.split(train_x):
    # print('train_index:%s , test_index: %s ' % (train_index, test_index))
    trainfold_x = train_x[train_index]
    trainfold_y = train_y[train_index]
    fold += 1
    # 設定RandomForest參數
    model = RandomForestRegressor(n_estimators=1200,
                                  min_samples_split=2,
                                  min_samples_leaf=2,
                                  max_features='auto',
                                  oob_score=True,
                                  random_state=1)
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
my_submission = pd.DataFrame({
    'id': train_id,
    'random_forests': train_array[:]
})
my_submission.to_csv(
    '{}.csv'.format('../result/random_forests/train_random_forests'),
    index=False)

# 生成新的驗證集
total_price = 0
for i in range(fold):
    total_price += val_dic['price' + str(i + 1)]
avg_price = total_price // fold
val_dic['random_forests'] = avg_price
my_submission = pd.DataFrame(val_dic)
my_submission = my_submission.loc[:, ['id', 'random_forests']]
my_submission.to_csv(
    '{}.csv'.format('../result/random_forests/val_random_forests'),
    index=False)

print(
    'train_mean_absolute_error',
    int(mean_absolute_error(val_y, my_submission.loc[:, ['random_forests']])))

# 生成新的測試集
total_price = 0
for i in range(fold):
    total_price += test_dic['price' + str(i + 1)]
avg_price = total_price // fold
test_dic['random_forests'] = avg_price
my_submission = pd.DataFrame(test_dic)
my_submission = my_submission.loc[:, ['id', 'random_forests']]
my_submission.to_csv(
    '{}.csv'.format('../result/random_forests/test_random_forests'),
    index=False)
