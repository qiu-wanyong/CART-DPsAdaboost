# coding: utf-8

# In[1]:


# Python3实现基于CART分类器的AdaBoost算法——census-income
import numpy as np
import pandas as pd


# In[2]:


# 因为数据已经过存储在csv文件中，就不需要再爬虫获取了


#  训练数据文件路径
train_path = './census-income.csv'

#  预测数据文件路径
predict_path = './census-income-test.csv'


def handle_data(filepath, miss='fill'):  # 定义处理数据的函数
    data = pd.read_csv(r'%s' % filepath) # 读取文件路径
    data = data.replace('?', np.nan) # 替换空值符号
    #  处理缺失值
    if miss == 'del':  # 删除掉缺失值
        miss_data = data.dropna(how='any')
    else:
        miss_data = data.fillna(method='ffill')
    # 因为预测数据和训练数据的标识有些许不同，因此在这里统一
    miss_data['Money'] = [1 if '-' in hh else -1 for hh in miss_data['Money']]  # 类别标签：不高于50K是1，高于50K是-1
    return miss_data


# 数据集分割为训练集和测试集
def split_data(trdata, percent_test=0.3, sample_count=13000):  # 同样存在样本不均衡的，
    all_s = np.arange(len(trdata))
    select_s = all_s[trdata[:, -1] == 1]
    random_s = np.random.choice(select_s, sample_count)
    # 解决样本不均衡问题：这里采用下采样(将多数类的数量减少)
    df0 = trdata[random_s]
    df1 = trdata[trdata[:, -1] == -1]
    trdata = np.vstack((df0, df1))
    np.random.shuffle(trdata)
    # 样本数不均衡，对
    kfoldict = {}
    length = len(trdata)
    sign = int(length * percent_test)
    # 生成随机数组
    random_list = np.arange(length)
    np.random.shuffle(random_list)
    kfoldict[0] = {}
    kfoldict[0]['train'] = trdata[random_list[sign:]]
    kfoldict[0]['test'] = trdata[random_list[:sign]]
    # 在训练数据的最后一列添加该样本的编号
    s_sign = np.arange(len(kfoldict[0]['train'])).reshape(-1, 1)
    kfoldict[0]['train'] = np.hstack((kfoldict[0]['train'], s_sign))
    return kfoldict

# In[61]:

# 训练数据
train_data = handle_data(train_path).values

# In[62]:

# 分割后的数据
dt_data = split_data(train_data)
print(dt_data)

# 预测数据
predict_data = handle_data(predict_path).values
print(len(predict_data))

# 将处理好的数据写入新建的.csv文件
def data_to_csv(data, name):
    df = pd.DataFrame(data)
    df.to_csv('./%s_real.csv' % name, index=False)
    print('数据写入成功')

for k in dt_data[0]:
    data_to_csv(dt_data[0][k], k)

data_to_csv(predict_data, 'predict')