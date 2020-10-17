

import pandas as pd
import numpy as np
import pystan


levels = 8
number_of_samples = 50

intercepts = np.random.normal(0,5,levels)

slopes = np.random.normal(5,2,levels)

x = np.random.normal(10, 10, number_of_samples)

df = pd.DataFrame()

df['x'] = x
df['factor'] = np.random.choice(levels, number_of_samples)
df['true_slope'] = df['factor'].apply(lambda x: slopes[x])
df['true_intercept'] = df['factor'].apply(lambda x: intercepts[x])

df['y'] = df['true_intercept'] + df['x'] * df['true_slope'] + np.random.normal(0,40,number_of_samples)


model1 = pystan.StanModel('model1.stan')


fit = model1.sampling(data = {'N': len(df), 'x': df['x'], 'y': df['y']})


fit_out = fit.to_dataframe().melt()
fit_out = fit_out.groupby('variable').mean().reset_index()
fit_out = fit_out[fit_out['variable'].str.contains('meanvector')]
fit_out['num'] = fit_out['variable'].str[11:-1]
fit_out['num'] = fit_out['num'].astype(int)
fit_out.sort_values('num',inplace=True)

df['pred1'] = np.asarray(fit_out['value'])
1 - ((df['pred1'] - df['y']) ** 2).sum() / (df['y'] ** 2).sum()


model2= pystan.StanModel('model2.stan')

fit = model2.sampling(data = {'N': len(df), 'x': df['x'], 'y': df['y'], 'no_factors': levels, 'factor': df['factor'] + 1})
fit_out = fit.to_dataframe().melt()
fit_out = fit_out.groupby('variable').mean().reset_index()
fit_out = fit_out[fit_out['variable'].str.contains('meanvector')]
fit_out['num'] = fit_out['variable'].str[11:-1]
fit_out['num'] = fit_out['num'].astype(int)
fit_out.sort_values('num',inplace=True)

df['pred2'] = np.asarray(fit_out['value'])
1 - ((df['pred2'] - df['y']) ** 2).sum() / (df['y'] ** 2).sum()

model3= pystan.StanModel('model3.stan')

df_train = df.query('factor not in [6,7]')
df_pred = df.query('factor in [6,7]')

data = {'N': len(df_train), 'x': df_train['x'], 'y': df_train['y'], 'no_factors': levels, 'factor': df_train['factor'] + 1}
data.update({'N_pred': len(df_pred), 'x_pred': df_pred['x'], 'y_pred': df_pred['y'], 'factor_pred': df_pred['factor'] + 1})

fit = model3.sampling(data = data)