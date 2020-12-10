import numpy as np 
import tensorflow as tf
import pandas as pd


data_path = 'stocks_data_input.npy'
ticker_list = ['AAPL', 'XOM' ,'VMC', 'BA', 'AMZN', 'TGT', 'WMT', 'KO', 'UNH', 'JPM', 'STT', 'MSFT', 'VZ', 'XEL', 'SPG']


data = np.load(data_path)
ohlc_features_num = data.shape[0]
ticker_num = data.shape[1]
trading_days_captured = data.shape[2]


print('number of ohlc features : ' + str(ohlc_features_num))
print('number of stocks considered : ' + str(ticker_num))
print('number of trading days captured : ' + str(trading_days_captured))

equiweight_weights_stocks = np.array([np.array([1/(ticker_num)]*(ticker_num+1))])


num_filters_layer_1 = 2
num_filters_layer_2 = 20
kernel_size = (1, 3)


train_data_ratio = 0.6
training_steps = 0.6 * trading_days_captured
validation_steps = 0.2 * trading_days_captured
test_steps = 0.2 * trading_days_captured


training_batch_size = 40
beta_pvm = 5e-5  
num_trading_periods = 10

weight_vector_init = np.array(np.array([1] + [0] * ticker_num))
portfolio_value_init = 10040
weight_vector_init_test = np.array(np.array([1] + [0] * ticker_num))
portfolio_value_init_test = 10040
num_episodes = 4
num_batches = 40
equiweight_vector = np.array(np.array([1/(ticker_num + 1)] * (ticker_num + 1)))

epsilon = 0.9
adjusted_rewards_alpha = 0.1

l2_reg_coef = 1e-8
adam_opt_alpha = 9e-2
optimizer = tf.train.AdamOptimizer(adam_opt_alpha)
trading_cost = 1/15000
interest_rate = 0.025/300
cash_bias_init = 0.65