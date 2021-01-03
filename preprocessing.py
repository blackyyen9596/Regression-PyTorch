import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gc


def preprocessing():
    # load data
    train_data = pd.read_csv('../ntut-ml-2020-regression/train-v3.csv')
    val_data = pd.read_csv('../ntut-ml-2020-regression/valid-v3.csv')
    test_data = pd.read_csv('../ntut-ml-2020-regression/test-v3.csv')

    # category feature one_hot
    test_data['price'] = -1
    # data = pd.concat([train_data, test_data])
    data = pd.concat((train_data, val_data, test_data))
    cate_feature = ['sale_yr', 'sale_month', 'sale_day']
    for item in cate_feature:
        data[item] = LabelEncoder().fit_transform(data[item])
        item_dummies = pd.get_dummies(data[item])
        item_dummies.columns = [
            item + str(i + 1) for i in range(item_dummies.shape[1])
        ]
        data = pd.concat([data, item_dummies], axis=1)
    data.drop(cate_feature, axis=1, inplace=True)

    train_val = data[data['price'] != -1]
    test = data[data['price'] == -1]

    # 清理內存
    del data, val_data, test_data
    gc.collect()

    # 18個特徵
    # features = [
    #     'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    #     'waterfront', 'view', 'condition', 'grade', 'sqft_above',
    #     'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
    #     'sqft_living15', 'sqft_lot15'
    # ]
    # 將不需要訓練的資料剃除
    # 63個特徵
    # del_feature = ['id', 'price']
    # features = [i for i in train_val.columns if i not in del_feature]
    features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'grade', 'sqft_above',
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
        'sqft_living15', 'sqft_lot15', 'sale_month1', 'sale_month2',
        'sale_month4', 'sale_month5', 'sale_month6', 'sale_month9',
        'sale_month11', 'sale_month12', 'sale_yr1', 'sale_yr2'
    ]

    # 將資料劃分為訓練集與測試集
    train_x = train_val[features][:train_data.shape[0]]
    val_x = train_val[features][train_data.shape[0]:]
    test_x = test[features]
    train_y = train_val['price'][:train_data.shape[0]].astype(int).values
    val_y = train_val['price'][train_data.shape[0]:].astype(int).values

    return train_x, val_x, test_x, train_y, val_y