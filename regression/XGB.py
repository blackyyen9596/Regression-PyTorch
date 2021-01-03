import xgboost as xgb
from xgboost import XGBRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from preprocessing import preprocessing as pre
from to_csv import to_csv

# 讀取預處理資料
train_x, val_x, test_x, train_y, val_y = pre()

# 標準化
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
val_x = ss.transform(val_x)
test_x = ss.transform(test_x)

# 將資料轉換為xgb格式
trn_data = xgb.DMatrix(train_x, label=train_y)
val_data = xgb.DMatrix(val_x, label=val_y)
watchlist = [(trn_data, 'train'), (val_data, 'valid')]
val_x = xgb.DMatrix(val_x)
test_x = xgb.DMatrix(test_x)
# 設定參數
num_round = 100000
params = {
    'min_child_weight': 10.0,
    'learning_rate': 1000,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'num_boost_round': 700,
    'nthread': -1,
    'missing': 1,
    'seed': 2019,
}
my_model = xgb.train(params,
                     trn_data,
                     num_round,
                     watchlist,
                     verbose_eval=20,
                     early_stopping_rounds=50)
# 將驗證集丟入模型中進行預測
val_y_pred = my_model.predict(val_x)
# 將測試集丟入模型中進行預測
test_y_pred = my_model.predict(test_x)
# 輸出模型評估指標
print('r2_score', round(r2_score(val_y, val_y_pred), 4))
print('mean_squared_error', int(mean_squared_error(val_y, val_y_pred)))
print('mean_absolute_error', int(mean_absolute_error(val_y, val_y_pred)))

# 將結果寫入csv檔中
to_csv(id_read_path=r'../ntut-ml-2020-regression/test-v3.csv',
       save_path=r'../result/xgboost/blacky.csv',
       y_pred=test_y_pred)
