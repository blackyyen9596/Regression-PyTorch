from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

# 建立XGBRegressor模型
model = XGBRegressor(
    learning_rate=0.04,
    n_estimators=1200,
    # 先調整 3, 5, 7, 9
    # 如果 3 最好
    # 再調整 2, 3, 4
    max_depth=6,
    # 先調整 1, 3, 5
    # 如果 5 最好
    # 再調整 4, 5, 6
    min_child_weight=5,
    # default = 1, range:(0, 1]
    subsample=0.8,
    # default = 1, range:(0, 1]
    colsample_bytree=1,
    # default = 'reg:linear'
    objective='reg:linear',
    # default = 0
    reg_lambda=0.2,
    # default = 0
    reg_alpha=0)
# 訓練模型
model.fit(train_x,
          train_y,
          early_stopping_rounds=200,
          eval_set=[(val_x, val_y)],
          verbose=True)
# 將驗證集丟入模型中進行預測
val_y_pred = model.predict(val_x)
# 將測試集丟入模型中進行預測
test_y_pred = model.predict(test_x)
# 輸出模型評估指標
# print('r2_score', round(r2_score(val_y, val_y_pred), 4))
# print('mean_squared_error', int(mean_squared_error(val_y, val_y_pred)))
print('mean_absolute_error', int(mean_absolute_error(val_y, val_y_pred)))

# 將結果寫入csv檔中
to_csv(id_read_path=r'../ntut-ml-2020-regression/test-v3.csv',
       save_path=r'../result/xgboost/blacky.csv',
       y_pred=test_y_pred)
