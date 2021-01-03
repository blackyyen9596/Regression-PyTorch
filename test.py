import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd

from preprocessing import preprocessing as pre
from regression.to_csv import to_csv

# 設定 GPU
device = torch.device("cuda")

# 讀取 csv 檔案
train_data = pd.read_csv('./dataset/train.csv', index_col=0)
val_data = pd.read_csv('./dataset/val.csv', index_col=0)
test_data = pd.read_csv('./dataset/test.csv', index_col=0)

train_x = train_data.iloc[:, 0:5].values
train_y = train_data.price.values
val_x = val_data.iloc[:, 0:5].values
val_y = val_data.price.values
test_x = test_data.iloc[:, 0:5].values

# 標準化
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
val_x = ss.transform(val_x)
test_x = ss.transform(test_x)

# pca降維
# pca = PCA(n_components=32)
# train_x = pca.fit_transform(train_x)
# val_x = pca.transform(val_x)
# test_x = pca.transform(test_x)

# 切割資料
val_x = torch.tensor(val_x, dtype=torch.float).to(device)
test_x = torch.tensor(test_x, dtype=torch.float).to(device)

# 讀取模型與權重
model = r'./logs/model.pth'
model = torch.load(model)

# 使用 GPU 測試
model.to(device)

# 將驗證集丟入模型中進行預測
with torch.no_grad():
    val_y_pred = model(val_x).detach().cpu().numpy()
# val_y_pred = ss_y.inverse_transform(val_y_pred)
# 將測試集丟入模型中進行預測
with torch.no_grad():
    test_y_pred = model(test_x).detach().cpu().numpy()
# test_y_pred = ss_y.inverse_transform(test_y_pred)
# 輸出模型評估指標
print('r2_score:', round(r2_score(val_y, val_y_pred), 4))
# print('mean_squared_error', int(mean_squared_error(val_y, val_y_pred)))
print('mean_absolute_error:', int(mean_absolute_error(val_y, val_y_pred)))

# 將結果寫入csv檔中
to_csv(id_read_path=r'../ntut-ml-2020-regression/test-v3.csv',
       save_path=r'../result/dnn/stacking_26.csv',
       y_pred=test_y_pred[:, 0])
