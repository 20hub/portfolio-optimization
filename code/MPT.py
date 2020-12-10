
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


def ReadData(TypeOfData):
    CryptoCont = 'C:/Users/saich/Desktop/Information Visualization Proj/data_files/crypto_cont_data_filtered.csv'
    Stocks = 'C:/Users/saich/Desktop/Information Visualization Proj/data_files/stocks_data.csv'
    CryptoPath = 'C:/Users/saich/Desktop/Information Visualization Proj/DS-5500-Project-Portfolio-Optimization-Using-Deep-Reinforcement-Learning-master/data_files/Crypto_Data.csv'
    
    if TypeOfData == "Crypto":
        Data = pd.read_csv(CryptoPath)
        Ticker = "crypto_ticker"
        Timeperiod = 365
    elif TypeOfData == "Stocks":
        Data = pd.read_csv(Stocks)
        Ticker = "ticker"
        Timeperiod = 252
    elif TypeOfData == "CryptoCont":
        Data = pd.read_csv(CryptoCont)
        Ticker = "crypto_ticker"
        Timeperiod = 365
    else:
        print("Error")

    return Data, Ticker

TypeOfData = "CryptoCont"
Data, Ticker = ReadData(TypeOfData)

Data = Data[['Date', Ticker, 'Close']]


Data['Date'] = pd.to_datetime(Data['Date'])
Data['Date'] = Data['Date'].dt.date



if TypeOfData == "CryptoCont":
    Data = Data.groupby(['Date', Ticker])['Close'].agg('mean').reset_index()


Table = Data.pivot(index='Date',columns=Ticker,values='Close') 


ColNames = Table.columns.values


Returns = Table.pct_change()


plt.figure(figsize=(16, 8))
for Col in ColNames:
    plt.plot(Returns.index, Returns[Col],label=Col)
plt.ylabel('Returns')
plt.legend(loc='upper left')
plt.title("Variation in daily returns for cryptocurrency")

def EfficientFrontier(AvgReturns, Covariance, CountPortfolio, RFF):
    
    WeightHolder = []
    Res = np.zeros((3,CountPortfolio))
    for j in range(CountPortfolio):
        
        Wei = np.random.random(6)
        Wei /= np.sum(Wei)
        WeightHolder.append(Wei)
        
        
        Por_SD = np.sqrt(np.dot(Wei.T, np.dot(Covariance, Wei))) * np.sqrt(Timeperiod)
        Por_Return = np.sum(AvgReturns*Wei ) *Timeperiod
        
        
        Res[0,j] = Por_SD
        Res[1,j] = Por_Return
        Res[2,j] = (Por_Return - RFF) / Por_SD

    
    VolatilityIndexMin = np.argmin(Res[0])
    MinVolSD, MinVolReturn = Res[0,VolatilityIndexMin], Res[1,VolatilityIndexMin]
    MinVolAllocation = pd.DataFrame(WeightHolder[VolatilityIndexMin],index=Table.columns,columns=['Alloc'])
    MinVolAllocation.Alloc = [round(i*100,2)for i in MinVolAllocation.Alloc]
    MinVolAllocation = MinVolAllocation.T
    
    SharpeIndexMax = np.argmax(Res[2])
    SDMaxSharpe, RetMaxSharpe = Res[0,SharpeIndexMax], Res[1,SharpeIndexMax]
    SharpeAllocationMax = pd.DataFrame(WeightHolder[SharpeIndexMax],index=Table.columns,columns=['Alloc'])
    SharpeAllocationMax.Alloc = [round(i*100,2)for i in SharpeAllocationMax.Alloc]
    SharpeAllocationMax = SharpeAllocationMax.T
    
    print("*" * 90)
    print("Portfolio Allocation for Minimum volatility\n")
    print("Portfolio Return Annual:", round(MinVolReturn,2))
    print("Portfolio Volatility Annual", round(MinVolSD,2))
    print("\n")
    print(MinVolAllocation)
    
    print("*" * 90)
    print("Portfolio Allocation for maximum sharpe ratio\n")
    print("Portfolio Return Annual:", round(RetMaxSharpe,3))
    print("Portfolio Volatility Annual", round(SDMaxSharpe,3))
    print("\n")
    print(SharpeAllocationMax)
    
    plt.figure(figsize=(14, 7))
    plt.savefig("mpt.png")
    plt.scatter(Res[0,:],Res[1,:],c=Res[2,:],cmap='YlGnBu', marker='o', s=6, alpha=0.5)
    plt.colorbar()
    plt.scatter(SDMaxSharpe,RetMaxSharpe, marker='*',color='g',s=300, label='Portfolio with Max Sharpe')
    plt.scatter(MinVolSD,MinVolReturn, marker='*',color='r',s=300, label='Portfolio with Min Volatility')
    plt.title('Modern Portfolio Theory using Efficient Frontier Theory')
    plt.xlabel('Volatility Annual')
    plt.ylabel('Returns Annual')
    plt.legend(labelspacing=0.6)


CountPortfolio = 30000
RFF = 0.0178
AvgReturns = Returns.mean()
Covariance = Returns.cov()

# Main function call to generate efficient frontier theory visualization
EfficientFrontier(AvgReturns, Covariance, CountPortfolio, RFF)
