import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
pd.options.mode.chained_assignment = None


print('##############Q1################')
heartf_path = os.path.abspath('heart_failure_clinical_records_dataset.csv')
df_heart = pd.read_csv(heartf_path)

df_heart_cols3 = df_heart.iloc[ : ,6:9]
df_heart_cols1 =  df_heart.iloc[ : ,2]
df_heart_colsdeath =  df_heart.iloc[ : ,-1]

df_heart_cols = pd.concat([df_heart_cols1,df_heart_cols3,df_heart_colsdeath], axis = 1)

#Q1.1
#loading the data into 2 Pandas dataframes:
grouped = df_heart_cols.groupby(df_heart_cols.DEATH_EVENT)
df_heart0 = grouped.get_group(0)
df_heart1 = grouped.get_group(1)
# print(df_heart0)
# print(df_heart1)
print("File read for heart failure data")


df_heart0 = df_heart0.iloc[:,:-1]
corrMatrix0 = df_heart0.corr()
# print (corrMatrix0)
sns.heatmap(corrMatrix0, annot=True,linewidths=1)
plt.title("Correlation matrix of Heart Failure data for surviving patients(0)")
plt.xlabel("Clinical Features")
plt.ylabel("Clinical Features")
plt.savefig('Corrmatrix surviving patients(0)')
plt.show()


df_heart1 = df_heart1.iloc[:, :-1]
corrMatrix1 = df_heart1.corr()
# print (corrMatrix1)
sns.heatmap(corrMatrix1, annot=True,linewidths=1)
plt.title("Correlation matrix of Heart Failure data for deceased patients(1)")
plt.xlabel("Clinical Features")
plt.ylabel("Clinical Features")
plt.savefig('Corrmatrix deceased patients(1)')
plt.show()

print('##############Q2&3################')

i_lst = [1,2,3]

df_final = pd.DataFrame({'Model': ['y = ax + b','y = ax2 + bx + c','y = ax3 + bx2 + cx + d','y = a log x + b', 'log y = a log x + b']*1,
                          'SSE (death event=0)': [0]*5,
                          '(death event=1)': [0]*5,                     
                        })
# print(df_final.iloc[0:1,1:2])
# print(df_final.iloc[0:1,2])
# print(df_final.iloc[1:2,1:2])
# print(df_final.iloc[1:2,2])

#print(df_final)

def sse(act,pred):
    squared_errors = (act - pred) ** 2
    return np.sum(squared_errors)


def linear_regression(a,b,d):
    X_train ,X_test , Y_train , Y_test = train_test_split (a,b,test_size =0.5,shuffle = False)
    degree = d
    weights = np. polyfit (X_train,Y_train, degree )
    model = np. poly1d ( weights )
    predicted = model (X_test)
    return weights,predicted,Y_test

def plotGraph(y_test,predicted,regressorName):
    if max(y_test) >= max(predicted):
        my_range = (max(y_test))
    else:
        my_range = (max(predicted))
    plt.scatter(range(len(y_test)), y_test, color='blue')
    plt.scatter(range(len(predicted)), predicted, color='red')
    plt.title(regressorName)
    plt.show()
    return

for i in i_lst:
    X = df_heart0['platelets']
    y = df_heart0['serum_creatinine']
    weights,predicted,Y_test = linear_regression(X,y,i)
    squared_errors = sse(Y_test,predicted)
    df_final.iloc[(i-1):i,1:2] = squared_errors
    print("Weights for degree {} are : {} for Surviving patients " .format(i, weights) )
    print("SSE for degree {} is {} for Surviving patients" .format(i, squared_errors))
    plotGraph(Y_test, predicted, "Predicted(Red) vs Actual(Blue) values for Surviving patients with degree {}".format(i))
    
for i in i_lst:
    X = df_heart1['platelets']
    y = df_heart1['serum_creatinine']
    weights,predicted,Y_test = linear_regression(X,y,i)
    squared_errors = sse(Y_test,predicted)
    df_final.iloc[(i-1):i,2] = squared_errors
    print("Weights for degree {} are : {} for Deceased patients " .format(i, weights) )
    print("SSE for degree {} is {} for Deceased patients" .format(i, squared_errors))
    plotGraph(Y_test, predicted, "Predicted(Red) vs Actual(Blue) values for Deceased patients with degree {}".format(i))


X = df_heart0['platelets']
y = df_heart0['serum_creatinine']
X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5,shuffle = False)
weights = np. polyfit (np.log(X_train),Y_train, 1 )
model = np. poly1d ( weights )
predicted = model (np.log(X_test))
squared_errors = sse(Y_test,predicted)
df_final.iloc[3:4,1:2] = squared_errors
print("Weights for degree y = a log x + b are : {} for Surviving patients " .format(weights) )
print("SSE for y = a log x + b is {} for Surviving patients" .format(squared_errors))
plotGraph(Y_test, predicted, "Predicted(Red) vs Actual(Blue) values for Surviving patients with y = a log x + b")


X = df_heart1['platelets']
y = df_heart1['serum_creatinine']
X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5,shuffle = False)
weights = np. polyfit (np.log(X_train),np.log(Y_train), 1 )
model = np. poly1d ( weights )
predicted = model (np.log(X_test))
squared_errors = sse(Y_test,predicted)
df_final.iloc[3:4,2] = squared_errors
print("Weights for degree logy = alogx + b are : {} for Deceased patients " .format(weights) )
print("SSE for logy = alogx + b is {} for Deceased patients" .format(squared_errors))
plotGraph(Y_test, predicted, "Predicted(Red) vs Actual(Blue) values for Deceased patients with logy = alogx + b")


X = df_heart0['platelets']
y = df_heart0['serum_creatinine']
X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5,shuffle = False)
weights = np. polyfit (np.log(X_train),np.log(Y_train), 1 )
model = np. poly1d ( weights )
predicted = model (np.log(X_test))
squared_errors = sse(Y_test,predicted)
df_final.iloc[4:5,1:2] = squared_errors
print("Weights for degree logy = a log x + b are : {} for Surviving patients " .format(weights) )
print("SSE for logy = alog x + b is {} for Surviving patients" .format(squared_errors))
plotGraph(Y_test, predicted, "Predicted(Red) vs Actual(Blue) values for Surviving patients with logy = alogx + b")

X = df_heart1['platelets']
y = df_heart1['serum_creatinine']
X_train ,X_test , Y_train , Y_test = train_test_split (X,y,test_size =0.5,shuffle = False)
weights = np. polyfit (np.log(X_train),Y_train, 1 )
model = np. poly1d ( weights )
predicted = model (np.log(X_test))
squared_errors = sse(Y_test,predicted)
df_final.iloc[4:5,2] = squared_errors
print("Weights for degree y = alogx + b are : {} for Deceased patients " .format(weights) )
print("SSE for y = alogx + b is {} for Deceased patients" .format(squared_errors))
plotGraph(Y_test, predicted, "Predicted(Red) vs Actual(Blue) values for Deceased patients with y = alogx + b")


#Creating the Dataframe for table structure as mentioned in question 3
print(df_final)  
df_final.to_csv('df_final.csv',index =False)

print(df_final.iloc[:,1])
print(df_final[1])