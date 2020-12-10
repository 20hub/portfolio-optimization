# 1.Overview
Financial portfolio optimization is the process of redistributing funds into multiple financial vehicles at a given timestamp to maximize returns while minimizing the risk at the same time.For a holding period of one trading day, we use our portfolio optimization code to construct weights of the 16 assets held in our portfolio.Then we rebalance the equity portion of our portfolio every trading day and keep using rolling window to update our method.

# 2.Framework
The baseline framework is implemented using the concept:

- Modern Portfolio Theory.
- The Reinforcement Learning framework is implemented using two machine learning methods: Convolutional Neural Network (CNN) and Long Short Term Memory (LSTM).

# 3.Set a Reinforcement Learning Environment
We propose a code file [RL Environment](code/RLEnvironment.ipynb) to set reinforcement learning trading environment.

The main factors are listed below:

- PortfolioValue: value of the finance portfolio
- TransCost: Transaction cost that has to be paid by the agent to execute the action
- ReturnRate: Percentage change in portfolio value
- WindowSize: Number of trading periods to be considered
- SplitSize: % of data to be used for training dataset, rest will be used for test dataset

# 4.Get Stocks and Cryptocurrency Trading data
We propose a code file [Get_data](code/data_scraping.ipynb) to get stocks and cryptocurrency trading data and build them into pandas dataframe.

The main sources are:

- Yahoo Finance
- NASDAQ

# 5.Data Analysis
After all the preparation, we propose a code file [Data_Analysis](code/exploratory_data_analysis.ipynb) to do the detailed analysis and iterations.

# 6.Other files
Other files regarding cnn-policy and LSTM-poly and their realization and the resulting parameters are listed as python files collected in [code] folder.
