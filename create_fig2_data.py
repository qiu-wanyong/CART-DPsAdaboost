#  根据前面的数据随机选择数据

train_count = 3000

test_count = 1000


predict_count = 3000

import pandas as pd


def random_select(filename, count, outname, c='normal'):
    data = pd.read_csv(filename)
    s_data = data.sample(n=count, replace=False)
    if c == 'train':
        # 需要重写最后一列的数据
        column_keys = list(s_data.keys())[-1]
        s_data[column_keys] = list(range(count))

    s_data.to_csv(outname, index=False)

    print('数据选取完毕')


# 预测数据
random_select('predict_real.csv', predict_count, 'predict_real_fig2.csv')


# 测试数据
random_select('test_real.csv', test_count, 'test_real_fig2.csv')


# 训练数据
random_select('train_real.csv', train_count, 'train_real_fig2.csv', 'train')