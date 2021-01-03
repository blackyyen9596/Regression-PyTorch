import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv('../ntut-ml-2020-regression/train-v3.csv')
train = train.drop([
    'id',
    'sale_day',
], axis=1)
train_y = train['price']

# one hot encoding
cate_feature = ['sale_month', 'sale_yr']
for item in cate_feature:
    le = LabelEncoder()
    train[item] = le.fit_transform(train[item])
    item_dummies = pd.get_dummies(train[item])
    item_dummies.columns = [
        item + str(i + 1) for i in range(item_dummies.shape[1])
    ]
    train = pd.concat([train, item_dummies], axis=1)
train.drop(cate_feature, axis=1, inplace=True)

# 計算訓練集整體相關係數，並繪製成熱像圖
corr = train.corr()
sns.heatmap(corr, xticklabels=False, yticklabels=False)
# plt.show()

# 查看跟價錢相關的係數
print(corr['price'])

# 刪除 price
train = train.drop(['price'], axis=1)

# 設定模型
estimator = LinearRegression()

# 原始特徵 + Logistic Regression
ss = StandardScaler()
train_x = ss.fit_transform(train)
org_cross_val_score = -cross_val_score(estimator,
                                       train_x,
                                       train_y,
                                       scoring='neg_mean_absolute_error',
                                       cv=10,
                                       verbose=1).mean()
print('org_cross_val_score:', round(org_cross_val_score, 4))

# 取出相關係數高於0.1或小於-0.1的行
Threshold = 0.005
high_list = list(corr[(corr['price'] > Threshold) |
                      (corr['price'] < Threshold * -1)].index)
high_list.pop(0)
# 篩選後的特徵
print(high_list)

# 篩選後特徵 + Logistic Regression
ss = StandardScaler()
train_x = ss.fit_transform(train[high_list])
change_cross_val_score = -cross_val_score(estimator,
                                          train_x,
                                          train_y,
                                          scoring='neg_mean_absolute_error',
                                          cv=10,
                                          verbose=1).mean()
print('change_cross_val_score:', round(change_cross_val_score, 4))
print('原始特徵數量:', train.shape[1])
print('篩選後特徵數量:', len(high_list))
print('分數變化:', round((org_cross_val_score - change_cross_val_score), 4))
