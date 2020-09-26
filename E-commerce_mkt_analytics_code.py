#!/usr/bin/env python
# coding: utf-8

# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import prettytable as PrettyTable
import matplotlib.pylab as plt
import os,sys
#module_path = os.path.abspath(os.path.join('../modules/'))
#if module_path not in sys.path:
#    sys.path.append(module_path)
#import boxcox
#import testing
import plumbing
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 15,8
#DATA_DIR = './data/'


# # Time Series Analytics

# In[109]:


#dateparse = lambda dates: pd.datetime.strptime(created_on, '%Y-%m-%d')
df1 = pd.read_csv('sales.csv')
df1.head(5)


# In[110]:


df1['shift'] = df1.salesday.shift(1).astype(np.datetime64)
#df1['shift2'] = df1.salesday.shift(-1).astype(np.datetime64)
df1['salesday1'] = df1['salesday'].astype(np.datetime64)


# In[119]:


df1.tail()


# In[120]:


df1['diff'] = df1['salesday1'] - df1['shift']
#df1['diff2'] = df1['salesday1'] - df1['shift2']


# In[132]:


df2=df1.loc[df1['salesday1'] >= '2016-01-01' ]


# In[133]:


len(df2)


# In[134]:


df3 = df2.drop(columns=['salesday','buyers','shift','diff'])


# In[135]:


df3.head()


# In[136]:


df3.set_index('salesday1').head()


# In[213]:


df3['total_amount'].plot()
plt.title("Sales\n2016 - 2017")
plt.show()


# In[142]:


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df3['total_amount'], model='multiplicative',freq = 30)
result.plot()
plt.show()


# In[145]:


from sklearn.model_selection import train_test_split  #splite the data into 80/20
train,test = train_test_split(df3['total_amount'], test_size=0.2, shuffle=False)
print('train: ',len(train))
print('test : ',len(test))

plt.plot(train, c='b', label='train')
plt.plot(test, c='r', label='test')
plt.legend()
plt.title('\n80/20 In Sample / Out of Sample')
plt.show()


# In[146]:


Y = np.log(df3['total_amount']).tolist()


# In[147]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[150]:


def plot_and_undo_boxcox(bc, Y_hat, method, in_sample=True):
    if in_sample:    Y = bc.Y
    else:            Y = bc.get_test()
    Y_plain = list(map(lambda y: bc.unapply(y), Y))
    Y_hat_plain = list(map(lambda y: bc.unapply(y), Y_hat))
    plt.plot(Y_hat_plain, label='Y_hat')
    plt.plot(Y_plain, label='Y')
    switch = {True: 'In',False: 'Out of'}
    plt.title('Predicted vs. Actual (%s Sample) \nAir Passengers\nUsing %s' % (switch[in_sample], method))
    plt.legend()
    plt.show()
    return (Y_plain, Y_hat_plain)


# In[152]:


class BoxCox(object):
    def __init__(self, X, test_size):
        assert min(X) > 0, 'only positive values can be transformed'
        self.X = X
        self.train, self.test = train_test_split(X, test_size=test_size, shuffle=False)
        try:
            self.Y, self.lbda = stats.boxcox(self.train)
        except:
            print('WARNING: unable to transform the data')
            self.Y, self.lbda = (self.train, None)
        print('Sampling       :  %d / %d\nBox Cox lambda : %s' % (len(self.Y), len(X), self.lbda))
    
    def plot(self):
        fig = plt.figure()
        ax_1 = fig.add_subplot(211)
        #X = air['Passengers'].tolist()
        prob = stats.probplot(self.train, dist=stats.norm, plot=ax_1)
        ax_1.set_title('Probability vs Normal\nRaw Data')
        ax_1.axes.get_xaxis().set_visible(False)
        
        ax_2 = fig.add_subplot(212)
        prob = stats.probplot(self.Y, dist=stats.norm, plot=ax_2)
        ax_2.set_title('After Box-Cox transformation')
        plt.show()
        
        header = ['data','mean','st_dev','std to mean']
        pt = PrettyTable(header)
        pt.add_row(['Raw', np.mean(self.train), np.std(self.train), np.std(self.train) / np.mean(self.train)])
        if self.lbda is not None:
            print('\nBox Cox Transform lambda : %f'% self.lbda)
            pt.add_row(['Box Cox', np.mean(self.Y), np.std(self.Y), np.std(self.Y) / np.mean(self.Y)])
        else:
            print('no tranformation!')
        print(pt)
    
    def get_test(self):
        return list(map(lambda x: self.apply(x), self.test))
    
    def apply(self, x):
        if self.lbda is None:
            return x
        elif self.lbda == 0:
            return np.log(x)
        else:
            return (x ** self.lbda - 1) / self.lbda
    
    def unapply(self, y):
        if self.lbda is None:
            return y
        elif self.lbda == 0:
            return np.exp(y)
        else:
            return (self.lbda * y + 1) ** (1/self.lbda)


# In[155]:


bc = BoxCox(df3['total_amount'], .2)


# In[158]:


bc.plot()


# ## Model One: Fourier Series

# In[160]:


X_f = []
for n,a in enumerate(bc.Y):
    x = [1, n, np.sin(2 * n * np.pi / 7), np.cos(2 * n * np.pi / 7)]
    X_f +=[x]


# In[161]:


reg = sm.OLS(bc.Y, X_f)
results = reg.fit()
results.summary()


# In[162]:


Y_hat = results.predict(X_f)
(Y_plain, Y_hat_plain) = plot_and_undo_boxcox(bc, Y_hat, 'Fourier Series', in_sample=True)


# In[163]:


i = len(bc.train)
X_pred = []
for n,(d,p) in enumerate(bc.test.items()):
    m = n+i
    x = [1, m, np.sin(2 * m * np.pi / 7), np.cos(2 * m * np.pi / 7)]
    X_pred += [x]
X_pred[:4]


# In[164]:


Y_pred = results.predict(X_pred)
(Y_plain, Y_hat_plain) = plot_and_undo_boxcox(bc, Y_pred, 'Fourier Series', in_sample=False)


# In[169]:


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[172]:


rms_1 = sqrt(mean_squared_error(bc.test,bc.unapply(Y_pred)))
print(rms_1)


# ## Model Two: Holt Winter Model

# In[173]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
holt = ExponentialSmoothing(bc.Y, trend='additive',seasonal='mul',seasonal_periods=7) 
result = holt.fit()


# In[174]:


Y_hat = result.fittedvalues
(Y_plain, Y_hat_plain) = plot_and_undo_boxcox(bc, Y_hat, 'Holt Winters', in_sample=True)


# In[175]:


Y_pred = result.forecast(len(bc.test))
(Y_plain, Y_hat_plain) = plot_and_undo_boxcox(bc, Y_pred, 'Holt Winters', in_sample=False)


# In[177]:


rms_2 = sqrt(mean_squared_error(bc.test,bc.unapply(Y_pred)))
print(rms_2)


# Holt winter is better!

# # Funnel Analytics

# In[191]:


from plotly import graph_objects as go


# In[192]:


#from plotly import graph_objects as go
fig = go.Figure(go.Funnel(
    y = ["Awareness", "Acquisition", "Purchase"],
    x = [5669, 1993, 1449],
    textposition = "inside",
    textinfo = "value+percent initial",
    opacity = 0.65, marker = {"color": ["deepskyblue", "tan", "teal", "silver"],
    "line": {"width": [4, 3, 1, 1], "color": ["wheat", "blue", "wheat", "wheat"]}},
    connector = {"line": {"color": "royalblue", "dash": "dot", "width": 3}})
    )

fig.show()


# # Customer Segmentation Analytics

# In[216]:


ac1 = pd.read_csv('mldataset.csv')
ac1.head(15)


# In[224]:


len(ac2)


# In[217]:


ac2 = ac1.drop(columns=['campaign','device_id'])


# In[218]:


ac2 = pd.get_dummies(ac2,columns=["device_type","operating_system"])


# In[219]:


ac2.head()


# In[220]:


X = ac2.drop(columns=['won'])


# In[221]:


Y = ac2.won


# In[208]:


import sklearn.tree as tree
from IPython.display import Image  
import pydotplus


# In[227]:


dt = tree.DecisionTreeClassifier(max_depth = 2)
dt.fit(X,Y)


# In[228]:


dt_feature_names = list(X.columns)
dt_target_names = np.array(Y.unique(),dtype=np.str) 
tree.export_graphviz(dt, out_file='tree.dot', 
    feature_names=dt_feature_names, class_names=dt_target_names,
    filled=True)  
graph = pydotplus.graph_from_dot_file('tree.dot')
Image(graph.create_png())


# In[ ]:




