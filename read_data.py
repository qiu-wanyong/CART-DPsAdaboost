
import pandas as pd


dt_data = {}
dt_data[0] = {}


train_data = pd.read_csv('./train_real.csv')


dt_data[0]['train'] = train_data.values

test_data = pd.read_csv('./test_real.csv')

dt_data[0]['test'] = test_data.values

predict_data = pd.read_csv('./predict_real.csv').values

print(dt_data)