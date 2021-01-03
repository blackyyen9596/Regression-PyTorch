from sklearn.preprocessing import StandardScaler
import sklearn.ensemble as se
import sklearn.tree as st
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
model.fit(train_x, train_y)
# 將驗證集丟入模型中進行預測
val_y_pred = model.predict(val_x)
# 將測試集丟入模型中進行預測
test_y_pred = model.predict(test_x)
# 輸出模型評估指標
# print('r2_score:', round(r2_score(val_y, val_y_pred), 4))
# print('mean_squared_error', int(mean_squared_error(val_y, val_y_pred)))
print('mean_absolute_error:', int(mean_absolute_error(val_y, val_y_pred)))

# 將結果寫入csv檔中
to_csv(id_read_path=r'../ntut-ml-2020-regression/test-v3.csv',
       save_path=r'../result/adaboost/blacky.csv',
       y_pred=test_y_pred)
