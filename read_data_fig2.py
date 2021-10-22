
import pandas as pd


dt_data_fig2 = {}
dt_data_fig2[0] = {}


train_data_fig = pd.read_csv('./train_real_fig2.csv')


dt_data_fig2[0]['train'] = train_data_fig.values

test_data_fig = pd.read_csv('./test_real_fig2.csv')

dt_data_fig2[0]['test'] = test_data_fig.values

predict_data_fig2 = pd.read_csv('./predict_real_fig2.csv').values
