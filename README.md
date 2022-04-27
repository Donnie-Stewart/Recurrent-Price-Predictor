<a href="https://colab.research.google.com/github/Donnie-Stewart/Recurrent-Price-Predictor/blob/main/Recurrent_Price_Predictor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
%matplotlib inline
```

# Recurrent Price Predictor

The purpose of the project is to explore using various Recurrent Neural Networks (RNN) supported by tensorflow, on tabular time series data. Specifically, the task at hand is to predict the next day's high price of Bitcoin based on prior closing date prices. Additionally, I will compare types of RNNs, discussing advantages of each model while implementing them. 

##Bitcoin Price Prediction
---
Bitcoin, amongst other assets like AMC and GME, has been all the rage this past year and reached an all time high of \$66,974 per bitcoin. Analysts continue to feed the frenzy by releasing price predictions that range from from \$500,000 to \$9,000 per bitcoin in the next year. I will implement a Recurrent Neural Network model to gain some insight into price prediction. [Yahoo! Finance](https://finance.yahoo.com/quote/BTC-USD/history/ ) is a trusted name in free financial information which I'll be using to gather data to train the RNNs.

The following code cells download the dependencies & training/test data. 


```python
!pip install yfinance
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
```


```python
#downloads BTC chart today - can input any valid date 
data = yf.download("BTC-USD", start="2014-09-15", end="2022-01-01")
#for saving and reformatting. 
data.to_csv('BTC-USD.csv')
data = pd.read_csv('BTC-USD.csv')
```


```python
data.head()
```





  <div id="df-a7d408b0-49fa-4b60-bf0a-61a0397efe10">
    <div class="colab-df-container">
      <div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-09-17</td>
      <td>465.864014</td>
      <td>468.174011</td>
      <td>452.421997</td>
      <td>457.334015</td>
      <td>457.334015</td>
      <td>21056800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-09-18</td>
      <td>456.859985</td>
      <td>456.859985</td>
      <td>413.104004</td>
      <td>424.440002</td>
      <td>424.440002</td>
      <td>34483200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-09-19</td>
      <td>424.102997</td>
      <td>427.834991</td>
      <td>384.532013</td>
      <td>394.795990</td>
      <td>394.795990</td>
      <td>37919700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-09-20</td>
      <td>394.673004</td>
      <td>423.295990</td>
      <td>389.882996</td>
      <td>408.903992</td>
      <td>408.903992</td>
      <td>36863600</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-09-21</td>
      <td>408.084991</td>
      <td>412.425995</td>
      <td>393.181000</td>
      <td>398.821014</td>
      <td>398.821014</td>
      <td>26580100</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a7d408b0-49fa-4b60-bf0a-61a0397efe10')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  
  </div>




Plotting the Bitcoin price against dates visualizes the data for this project. The goal is to predict the next day's high price of Bitcoin based on a predetermined look back date. Visualizing can help provide a sanity check to determine how close the model is predicitng to the actual model.   


```python
l = len(data)
g = sns.lineplot(x = np.linspace(1,l,l), y = data['High'].values.reshape(-1))
g.set(xticks=np.arange(0,l,200))
g.set_xticklabels(rotation=30, labels = data['Date'][0::200])
```




    [Text(0, 0, '2014-09-17'),
     Text(0, 0, '2015-04-05'),
     Text(0, 0, '2015-10-22'),
     Text(0, 0, '2016-05-09'),
     Text(0, 0, '2016-11-25'),
     Text(0, 0, '2017-06-13'),
     Text(0, 0, '2017-12-30'),
     Text(0, 0, '2018-07-18'),
     Text(0, 0, '2019-02-03'),
     Text(0, 0, '2019-08-22'),
     Text(0, 0, '2020-03-09'),
     Text(0, 0, '2020-09-25'),
     Text(0, 0, '2021-04-13'),
     Text(0, 0, '2021-10-30')]




![png](Recurrent_Price_Predictor_files/Recurrent_Price_Predictor_9_1.png)


### A) Data Preprocessing
The 5 columns displayed above won't all be necessesary for this project. For predicting the next day's High I will consider only prior daily Highs. 


```python
data_high = data['High'].to_numpy()
data_high = data_high.reshape(-1, 1)
```


[MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) from sklearn will be used to reformat the data. The current range is from less than \$1 to greater than \$60,000. After scaling, the data is proportionally the same but is in the range 0 to 1. Will need to remember this function later for reverse scaling when predicting. 
- Side note: This is an example of data normalization, preventing any individual number from being overly influential to the system. 


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_high)
data_normalized = scaler.transform(data_high)
```

I will implement a simple autoregressive recurrent neural network using the standard tensorflow RNN architectures. An autoregressive model originates from the literature on time-series models where observations from the previous time-steps are used to predict the value at the current time step. To implemement an autoregressive model, I will simply augment the data so that "time_steps", referring to the number of previous days, are fed to the model at the current time step in order to form a prediction. [Here](https://www.exxactcorp.com/blog/Deep-Learning/recurrent-neural-networks-rnn-deep-learning-for-sequential-data) is more information on autoregressive modeling as well as a justification for LSTM. 


```python
def create_dataset(dataset, time_steps=1):
    """
    Generate a dataset where the time series X[n] contains the readings for the 'time_step' previous days 
    and y contains the price for today.
    args:
    dataset: numpy array, the data
    time_steps: the number of previous days to feed to RNN

    returns:
    tuple: a dataset with x[i] containing 'time_step' number of previous prices, target price for x[i]
    """
    dataX, dataY = [],[]
    for i in range(len(dataset)-time_steps-1):
        a = dataset[i:(i+time_steps)]
        dataX.append(a)
        dataY.append(dataset[i + time_steps])
    return np.array(dataX), np.array(dataY)
```


```python
# Produces a dataset based on the number of days the model could look back
time_steps = 20
X, y = create_dataset(data_normalized, time_steps)
```


```python
# Check the shape of your dataset; should be (n, time_steps, 1) and (n, 1)
#i.e. n previous data is input -  y is the following day
print(X.shape, y.shape)
```

    (2643, 20, 1) (2643, 1)


### B) Data Partitioning
Split data into train and test sets. Using 80\% for training and 20\% for testing. 


```python
split = ((X.shape[0])*8)//10
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (2114, 20, 1) (529, 20, 1) (2114, 1) (529, 1)


For this dataset, I needed to reshape the partitions for the model to be able to process them.


```python
# Reshape input to be [samples, features, timesteps].
X_train = np.reshape(X_train, (X_train.shape[0],1,  X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (2114, 1, 20) (529, 1, 20) (2114, 1) (529, 1)


###  C) RNN Model 1: LSTM

In this part I will create a model using an RNN layer (Specifically LSTM) and train it on the training data. I will also plot training and validation loss using mean squared error as your model's metric.


```python
# Build model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
lstm = tf.keras.models.Sequential([
    LSTM(units = 32, activation = 'relu' , return_sequences = True),     #recurrent activation sigmoid, output activation tanh
    LSTM(units = 64, activation = 'relu' ,return_sequences = True),
    LSTM(units = 128, activation = 'relu' ),
    Dense(units=1)
])
# lstm.summary()
loss = "mean_squared_error"
opt = "adam"
metrics = [MeanSquaredError(), MeanAbsoluteError()]
lstm.compile(optimizer=opt, loss= loss, metrics = metrics)
```


```python
batchsize = 32
epochs =  30
# Fit model
history = lstm.fit(X_train, y_train, validation_split = .2, epochs=epochs, batch_size=batchsize, verbose = 0)
```


```python
# Plot the Model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print(history.history['loss'])
print(history.history['val_loss'])
print("Final MSE training model loss is:", history.history['loss'][-1])
print("Final MSE validation model loss is:", history.history['val_loss'][-1])
```


![png](Recurrent_Price_Predictor_files/Recurrent_Price_Predictor_25_0.png)


    [0.0030514108948409557, 0.0005694805295206606, 0.00012279443035367876, 0.00011297076707705855, 9.749775927048177e-05, 8.421274833381176e-05, 7.769341755192727e-05, 6.515343557111919e-05, 5.352998778107576e-05, 4.7229368647094816e-05, 4.561291279969737e-05, 3.729785748873837e-05, 3.854155875160359e-05, 3.549851317075081e-05, 3.4562785003799945e-05, 3.26493609463796e-05, 3.417179323150776e-05, 3.1600149668520316e-05, 3.40892729582265e-05, 3.1422085157828405e-05, 2.892608245019801e-05, 3.0866875022184104e-05, 2.9167513275751844e-05, 2.977934491354972e-05, 2.8183132599224336e-05, 2.6782334316521883e-05, 2.6159646949963644e-05, 2.475340988894459e-05, 2.5765306418179534e-05, 2.475959263392724e-05]
    [0.005233852192759514, 0.0002445350692141801, 0.00014303038187790662, 0.00012845819583162665, 0.00011813497258117422, 0.00010581845708657056, 0.00012724049156531692, 8.468801388517022e-05, 7.111149170668796e-05, 0.00012086513015674427, 7.675852975808084e-05, 5.7660781749291345e-05, 6.638849299633875e-05, 6.013852544128895e-05, 5.58798601559829e-05, 6.661679071839899e-05, 5.562518344959244e-05, 5.673550913343206e-05, 5.513741780305281e-05, 5.2272178436396644e-05, 6.006096737110056e-05, 4.937082849210128e-05, 5.0787893997039646e-05, 6.953967385925353e-05, 4.793375774170272e-05, 5.070084080216475e-05, 5.028834493714385e-05, 4.627060116035864e-05, 4.557559805107303e-05, 6.853922241134569e-05]
    Final MSE training model loss is: 2.475959263392724e-05
    Final MSE validation model loss is: 6.853922241134569e-05


### D) RNN Model 2: GRU
In this part, I will create an RNN model that instead uses a similar structure, GRU, for comparison. 


```python
# Build model
from tensorflow.keras.layers import GRU, Dropout
gru = tf.keras.models.Sequential([
    GRU(units = 32, activation = 'relu' , return_sequences = True),  
    GRU(units = 64, activation = 'relu' ,return_sequences = True),
    GRU(units = 128, activation = 'relu'),
    Dense(units=1)
])
loss = "mean_squared_error"
opt = "adam"
metrics = [MeanSquaredError(), MeanAbsoluteError()]
gru.compile(optimizer=opt, loss= loss, metrics = metrics)
```


```python
batchsize = 32
epochs =  60

# Fit model
history = gru.fit(X_train, y_train, validation_split = .2, epochs=epochs, batch_size=batchsize, verbose = 0)
```


```python
# Plot the Model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print(history.history['loss'])
print(history.history['val_loss'])
print("Final MSE training model loss is:", history.history['loss'][-1])
print("Final MSE validation model loss is:", history.history['val_loss'][-1])
```


![png](Recurrent_Price_Predictor_files/Recurrent_Price_Predictor_29_0.png)


    [0.001333520282059908, 0.00011788093252107501, 0.00010595640924293548, 7.795543206157163e-05, 5.919947579968721e-05, 4.764654295286164e-05, 4.0866976632969454e-05, 4.154067573836073e-05, 3.451843076618388e-05, 3.393289080122486e-05, 3.117513551842421e-05, 3.1902258342597634e-05, 3.2383628422394395e-05, 3.066410863539204e-05, 2.7998039513477124e-05, 3.023591489181854e-05, 3.5119446692988276e-05, 3.207785266567953e-05, 2.9387842005235143e-05, 2.5864161216304637e-05, 2.6532094125286676e-05, 2.6615218303049915e-05, 2.7471291105030105e-05, 2.6809399059857242e-05, 2.4802433472359553e-05, 2.457727532600984e-05, 2.5655323042883538e-05, 2.2826221538707614e-05, 2.4886323444661684e-05, 2.276029044878669e-05, 2.5729726985446177e-05, 2.1029360141255893e-05, 2.7224681616644375e-05, 2.653030060173478e-05, 2.1401760022854432e-05, 2.173430402763188e-05, 2.2725362214259803e-05, 3.2898347853915766e-05, 2.2015508875483647e-05, 2.2062904463382438e-05, 2.1253128579701297e-05, 2.3191505533759482e-05, 2.210579077654984e-05, 2.2853204427519813e-05, 2.4777385988272727e-05, 2.4201164706028067e-05, 3.426723924349062e-05, 2.155558468075469e-05, 2.397420394117944e-05, 1.9291095668449998e-05, 1.8965611161547713e-05, 2.0306712031015195e-05, 1.7104328435380012e-05, 2.163825229217764e-05, 2.675311225175392e-05, 2.0535999283310957e-05, 1.929487734741997e-05, 1.809758941817563e-05, 1.742695712891873e-05, 2.0146231690887362e-05]
    [0.00020439713262021542, 0.00013208236487116665, 0.0001234442606801167, 9.362729178974405e-05, 9.474438411416486e-05, 7.586136052850634e-05, 0.00010181485413340852, 5.4386960982810706e-05, 5.429224984254688e-05, 5.3650768677471206e-05, 6.202368240337819e-05, 5.126036558067426e-05, 0.00011184716277057305, 5.96699865127448e-05, 0.00010919742635451257, 8.94222830538638e-05, 4.8037451051641256e-05, 5.114329542266205e-05, 4.6842778829159215e-05, 4.477362017496489e-05, 4.468056431505829e-05, 5.075701847090386e-05, 6.085088534746319e-05, 5.785911707789637e-05, 4.4432043068809435e-05, 4.523840470938012e-05, 4.287675255909562e-05, 5.207737922319211e-05, 4.163742414675653e-05, 4.4288277422310784e-05, 4.141265162616037e-05, 5.954016523901373e-05, 4.5324133679969236e-05, 5.6818775192368776e-05, 4.102790626347996e-05, 4.150012682657689e-05, 4.134251503273845e-05, 0.00011398351489333436, 5.106384924147278e-05, 4.0336693928111345e-05, 7.715268293395638e-05, 4.051310315844603e-05, 6.735524220857769e-05, 4.2866839066846296e-05, 3.8755089917685837e-05, 4.2222265619784594e-05, 4.398381133796647e-05, 3.7764129956485704e-05, 3.788879985222593e-05, 4.2117037082789466e-05, 5.3222422138787806e-05, 3.730784737854265e-05, 3.677840868476778e-05, 4.961473314324394e-05, 4.16868097090628e-05, 4.1924235119950026e-05, 3.716721766977571e-05, 4.359244849183597e-05, 3.550694236764684e-05, 4.628036549547687e-05]
    Final MSE training model loss is: 2.0146231690887362e-05
    Final MSE validation model loss is: 4.628036549547687e-05


### E) Looking at the Predictions

Now, I will display my best model's performance on the test set, plotting the model's prediction for Bitcoin Price along with the actual test set prices. 

**Note:** The model is trained on normalized data and thus predicts a normalized score. In order to transform the model's predictions to the original price range will have to use sklearn's [inverse_transform](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)


```python
# y_pred_gru =  scaler.inverse_transform(gru.predict(X_test)) 
# y_pred_lstm = scaler.inverse_transform(lstm.predict(X_test))
y_pred_avg =  (scaler.inverse_transform(gru.predict(X_test)) +scaler.inverse_transform(lstm.predict(X_test)))/2
y_test2 =  scaler.inverse_transform(y_test)
# plt.plot(y_pred_gru)
# plt.plot(y_pred_lstm)
plt.plot(y_pred_avg)
plt.plot(y_test2)
plt.title('model loss on actual BTC')
plt.ylabel('price')
plt.xlabel('date')
plt.legend([ 'y_pred_avg', 'y_test'], loc='upper left')
plt.show()
```


![png](Recurrent_Price_Predictor_files/Recurrent_Price_Predictor_31_0.png)


After testing lookback days , layer amount, dropout, and different architectures, I (by coincidence) made a GRU that undershoots the price and an LSTM that overshoots it, and taking the average of their prediction proved to be closest to the actual bitcoin prices. 20 look back days was empirically the best for my architectures. It potentially was never the right complexity when I tested a higher amount of look back days (up to 100). Layer amount of three seemed to be the best for both the GRU and LSTM that I tested. The GRU usually provided lower losses but because these models were working with fractional squared loss, the LSTM may have performed better with other metrics like absolute error. Thus the best model I found was actually a combination of both GRU and LSTM. 


