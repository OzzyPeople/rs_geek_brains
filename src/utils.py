import pandas as pd
import numpy as np


def prefilter_items(data, item_features, take_n_popular=5000):
    """Предфильтрация товаров"""

    # определим цену
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))

    # определим stm
    private_label = [x for x in item_features[item_features['brand'] == 'Private'].item_id.unique()]
    data['stm'] = 0
    data.loc[data['item_id'].isin(private_label), 'stm'] = 1

    # 1. Удаление товаров, со средней ценой < 1$
    price_mean = data.groupby('item_id')['price'].mean().reset_index()
    less_1 = [x for x in price_mean['item_id'].loc[price_mean['price'] < 1]]
    data = data.loc[~data['item_id'].isin(less_1)]

    # 2. Удаление товаров со средней ценой > 30$
    more_1 = [x for x in price_mean['item_id'].loc[price_mean['price'] > 30]]
    data = data.loc[~data['item_id'].isin(more_1)]

    # 3. Придумайте свой фильтр - товары, которые не продавались последние 6 мес
    notsold = data['item_id'][(data['week_no'] <= 6 * 4) & (data['sales_value'] == 0)]
    data = data[~data['item_id'].isin(notsold)]

    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popularity_sum = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity_sum.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_5000 = popularity_sum.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    data = data.loc[data['item_id'].isin(top_5000)]
    # your_code

    return data

def postfilter_items(user_id, recommednations):
    pass