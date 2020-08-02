import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#from binarytree import Node

train_header=[]
total_rows=0
total_cols=0


class DecisionTree:
    
    def fun(self):  
      pass
    
    class Node: 
        def __init__(self,key): 
            self.left = None
            self.right = None
            self.val = key
    
    def remove_null(self,ds1):
        
        ds1.drop(ds1.iloc[:,0:1],inplace=True,axis=1)
        ds1.drop(ds1.iloc[:,79:80],inplace=True,axis=1)

        
        total_rows=int(ds1.shape[0])
        total_cols=int(ds1.shape[1]) 
        
        l=list(ds1.isnull().sum())
        to_delete=[]
        
        print(total_rows,total_cols)
        
        for i in range(total_cols):
            
            avg=int(l[i])/total_rows        
            if(avg>=0.70):
                to_delete.append(i)
        
        print(to_delete)
        return to_delete
        ds1.drop(ds1.columns[to_delete],inplace=True,axis=1)

    
        total_rows=int(ds1.shape[0])
        total_cols=int(ds1.shape[1])
                  
    def fill_missing(self,df1):
        for i in df1:
            if (df1[i].dtype!="object"):
                
                df1[i].fillna(df1[i].mean(),inplace=True)

            else:
                df1[i].fillna(df1[i].mode()[0],inplace=True)

    
    
    def cal_mean(self,a1):
        # a1.sort()
        # if (isinstance(a1[0],int) == False and isinstance(a1[0],float)==False):
        #     return a1
        
        # list=[]
        # for i in range(0,len(a1)-1):
        #     list.append(a1[i]+a1[i+1]/2)
        # return list
        return a1
    
    def cal_MSE(self,df,price):
        
        a1=pd.DataFrame(df).to_numpy().ravel()
        
        df=pd.concat([df,price],axis=1)
        
        mean_ar=self.cal_mean((set(a1.flatten()) ))
        
        min_mse=[float('inf'),0,0]
        
        for i in mean_ar:
            df1=df[ df.iloc[:,0] > i ]
            df2=df[ df.iloc[:,0] <= i ]
            
            if(df.iloc[:,0].dtype=="object"):
                df1=df[ df.iloc[:,0] == i ]
                df2=df[ df.iloc[:,0] != i ]
            
            Y_true=df1.iloc[:,1].to_numpy().ravel().flatten()
            Y_pred=df2.iloc[:,1].to_numpy().ravel().flatten()
            
            
            if(len(Y_true)==0 or len(Y_pred)==0):
                # print("-",end='')
                continue
    
            MSE1=0
    
            for j in Y_true:
                MSE1+=(Y_true.mean()-j)**2
            MSE2=0
            for j in Y_pred:
                MSE2+=(Y_pred.mean()-j)**2            
                
    
            MSE1=MSE1*len(Y_true)
            MSE2=MSE2*len(Y_pred)
    
            MSE=(MSE1+MSE2)/price.shape[0]
            
            if(min_mse[0]>MSE):
                min_mse[0]=MSE
                min_mse[1]=i
                
        return min_mse
            
    
    
    def find_MSE(self,df,price):
        list=[]
        for i in df:
            temp=self.cal_MSE(df[i],price)
            temp[2]=i
    #        print(i,end=' ')
    #        print(temp)
            list.append( temp)
        return list
    
    def build_tree(self,df,price,depth):
        
        
        print("p-",price.shape[0],end=' ')
        print("df-",df.shape[0],end='')
        if(df.shape[0]<=8 or df.shape[1]<=1 ):
            return None
        
        list=self.find_MSE(df,price)
        
        min=0
        
        for i in range(len(list)):
            if(list[i][0]<list[min][0]):
                min=i
        # print(list[min])        
        root=self.Node(list[min])
        
        df=pd.concat([df,price],axis=1)    
        
        
        df1=pd.DataFrame(columns=None)
        df2=pd.DataFrame(columns=None)
        
    
        if(df[list[min][2]].dtype=="object"):
            df1=df[ df[ list[min][2]] == list[min][1]]
            df2=df[ df[ list[min][2]] != list[min][1] ]  
        else:
            df1=df[ df[ list[min][2]] > list[min][1] ]
            df2=df[ df[ list[min][2]] <= list[min][1] ]        
    
        if(df1.shape[0]<=8 or df2.shape[0]<=8 or depth>5):
#          print(df1.shape[0],df2.shape[0])
#          print("drop at depth",depth)
          return root
    
#        print("\n\n",list[min], "\tspliting , depth == ",depth," shape ",df1.shape[0],df2.shape[0])         
        
        # del df1[list[min][2]]
        # del df2[list[min][2]]
        
        p1=df1["SalePrice"]
        p2=df2["SalePrice"]
        
    #    print(df.shape[1])
        del df1["SalePrice"]
        del df2["SalePrice"]    
        
        root.left=self.build_tree(df1,p1,depth+1)
        root.right=self.build_tree(df2,p2,depth+1)
        
        return root
    

    
    def printInorder(self,root): 
      
        if root: 
            self.printInorder(root.left) 
            print(root.val), 
            self.printInorder(root.right) 
    
    
    def cal_price(self,df):
        # print(df.shape[0])
        return df['SalePrice'].to_numpy().mean()
            
            
    def prediction(self,root,sample,df):
        # print(df.shape[0],end=' ')
        if(root==None):
            return self.cal_price(df)    
        # print(root.val,sample[root.val[2]])
        if(df.shape[0]<=8):
          return self.cal_price(df)
        
    
    
        if(root.left==None and root.right==None):
          return self.cal_price(df)
    
        
        if(sample[root.val[2]]=="object"):
            
            if(sample[root.val[2]] == root.val[1]):
                    
                return self.prediction(root.left,sample,df[ df[ root.val[2]] == root.val[1] ])
      
            elif(sample[root.val[2]] != root.val[1]):
                           
                           
                return self.prediction(root.right,sample,df[ df[ root.val[2]] != root.val[1] ])
        else:
            # print(sample[root.val[2]] , root.val[1])
            if(sample[root.val[2]] > root.val[1]):
    
                return self.prediction(root.left,sample,df[ df[ root.val[2]] > root.val[1] ])
        
            elif(sample[root.val[2]] <= root.val[1]):
                
                return self.prediction(root.right,sample,df[ df[ root.val[2]] <= root.val[1] ])
                           

    train_path=""
    null_list=[]
    root=0
    df=pd.DataFrame(columns=None)

    
    def train(self,p):
        self.root=self.Node(0)
        self.train_path=p
        df1=pd.read_csv(self.train_path,header=None)
        train_header=df1.iloc[0:1,:].to_numpy()[0]
        df1=pd.read_csv(self.train_path)
        train_ID=df1.iloc[:,0:1]
        
        train_labels=df1.iloc[:,80:81]        
        self.null_list=self.remove_null(df1)
        
        self.fill_missing(df1)        
        df1.drop(df1.columns[self.null_list],inplace=True,axis=1)
        
        self.fill_missing(df1)
        print(df1,train_labels)
        self.root=self.build_tree(df1.iloc[0:,:],train_labels[0:],0)
        
        self.df=pd.concat([df1,train_labels],axis=1)
        
        print("******************tree done*****************")        
#        print(p)    


    test_path=""
    def predict(self,p):
        self.test_path=p
        df2=pd.read_csv(self.test_path)
        df2.drop(df2.columns[self.null_list],inplace=True,axis=1)
        self.fill_missing(df2)
        
        rslt=[]
        
        for i in range(df2.shape[0]):
        #    print(df2.iloc[i,:])
            x=self.prediction(self.root,df2.iloc[i,:],self.df)
            rslt.append(x)         
        return rslt
#    df1=pd.read_csv("train.csv",header=None)
#    train_header=df1.iloc[0:1,:].to_numpy()[0]
#    
#    df1=pd.read_csv("train.csv")
#    df2=pd.read_csv("test.csv")
#    df3=pd.read_csv("test_labels.csv",header=None)
#    
#    train_ID=df1.iloc[:,0:1]
#    
#    train_labels=df1.iloc[:,80:81]
#    
#    remove_null(df1,df2)
#    fill_missing(df1,df2)
#    
#    #cal_MSE(df1.iloc[:,1],train_labels)
#    df=pd.concat([df1,train_labels],axis=1)
#    root=build_tree(df1.iloc[0:500,:],train_labels[0:500],0)
#    
#    print("******************tree done*****************")
#    
#    df=pd.concat([df1,train_labels],axis=1)
#    
#    rslt=[]
#    
#    for i in range(df2.shape[0]):
#    #    print(df2.iloc[i,:])
#        x=prediction(root,df2.iloc[i,:],df)
#        rslt.append(x)
#            
#    print (r2_score(rslt, df3[1].to_numpy()))    

#
#from q3 import DecisionTree as dtree
#dtree_regressor = DecisionTree()
#dtree_regressor.train('./Datasets/q3/train.csv')
#predictions = dtree_regressor.predict('./Datasets/q3/test.csv')
#test_labels = list()
#with open("./Datasets/q3/test_labels.csv") as f:
#  for line in f:
#    test_labels.append(float(line.split(',')[1]))
#print (mean_squared_error(test_labels, predictions))
#print (r2_score(test_labels, predictions))