import numpy as np
import pandas as pd

class RLEnv():
    
    """
    In this section, we build Rlenvironment.
    """
 
    def __init__(self, Path, PortfolioValue, TransCost, ReturnRate, WindowSize, TrainTestSplit):
       
        
        
        self.Dataset = np.load(Path)
        
      
        self.NumStocks = self.Dataset.shape[1]
        self.NumValues = self.Dataset.shape[0]
        
     
        self.PortfolioValue = PortfolioValue
        self.TransCost = TransCost
        self.ReturnRate = ReturnRate
        self.WindowSize = WindowSize
        self.Done = False
        
     
        self.state = None
        self.TimeLength = None
        self.Terminate = False
        
      
        self.TerminateRows = int((self.Dataset.shape[2] - self.WindowSize) * TrainTestSplit)
        
    def UpdatedOpenValues(self, T):
       
        return np.array([1+self.ReturnRate]+self.Dataset[-1,:,T].tolist())
    
    def InputTensor(self, Tensor, T):
        return Tensor[: , : , T - self.WindowSize:T]
    
    def ResetEnvironment(self, InitWeight, InitPortfolio, T):
        self.state= (self.InputTensor(self.Dataset, self.WindowSize) , InitWeight , InitPortfolio)
        self.TimeLength = self.WindowSize + T
        self.Done = False
        
        return self.state, self.Done
    
    def Step(self, Action):
       
        
   
        Dataset = self.InputTensor(self.Dataset, int(self.TimeLength))
    
    
       
        weight_vector_old = self.state[1]
        portfolio_value_old = self.state[2]
        
     
        NewOpenValues = self.UpdatedOpenValues(int(self.TimeLength))
        
        
        WeightAllocation = Action
        PortfolioAllocation = portfolio_value_old
        
        
        TransactionCost = PortfolioAllocation * self.TransCost * np.linalg.norm((WeightAllocation-weight_vector_old),ord = 1)
        
       
        ValueAfterTransaction = (PortfolioAllocation * WeightAllocation) - np.array([TransactionCost]+ [0] * self.NumStocks)
        
        
        NewValueofStocks = ValueAfterTransaction * NewOpenValues
        
        
        NewPortfolioValue = np.sum(NewValueofStocks)
        
        
        NewWeightVector = NewValueofStocks / NewPortfolioValue
        
        
        RewardValue = (NewPortfolioValue - portfolio_value_old) / (portfolio_value_old)

        self.TimeLength = self.TimeLength + 1
        
        
        self.state = (self.InputTensor(self.Dataset, int(self.TimeLength)), NewWeightVector, NewPortfolioValue)
        
        
        if self.TimeLength >= self.TerminateRows:
            self.Done = True
            
        return self.state, RewardValue, self.Done