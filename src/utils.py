from src.network import Network

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def split_dataset(train_df: pd.DataFrame, 
                  fraction: float = 0.2):

    #permute all samples
    train_df = train_df.sample(frac=1.0)

    #set validation fraction
    fraction = 0.2

    split_index = int(fraction*len(train_df))

    valid_df = train_df.iloc[:split_index]
    train_df = train_df.iloc[split_index:]

    return train_df, valid_df


def train_net( net: Network,
               dataset: pd.DataFrame,
               max_epochs: int,
               learning_rate: float,
               batch_size: int = 1,
               multiclass: bool = False):
    
    train_df, valid_df = split_dataset(dataset, 0.2)
    train_y_df = pd.get_dummies(train_df['cls'], dtype=float) if multiclass else train_df['cls']
    valid_y_df = pd.get_dummies(valid_df['cls'], dtype=float) if multiclass else valid_df['cls']

    y_dim = train_y_df.shape[1] if multiclass else 1

    train_losses = []
    validation_losses = []
    for epoch in range(max_epochs):
        train_loss = 0
        validation_loss = 0
        for i in range(0, len(train_df)-batch_size, batch_size):
            x = np.array(train_df.iloc[i:i+batch_size, :-1])
            y = np.reshape(np.array(train_y_df.iloc[i:i+batch_size, :]), (batch_size, y_dim))
            loss = net.fit(x, y, learning_rate, batch_size)
            train_loss +=loss

        for i in range(0, len(valid_df)-batch_size, batch_size):
            x = np.array(valid_df.iloc[i:i+batch_size, :-1])
            y = np.reshape(np.array(valid_y_df.iloc[i:i+batch_size, :]), (batch_size, y_dim))
            loss = net.validate(x, y, learning_rate, batch_size)
            validation_loss +=loss

        train_losses.append(train_loss/len(train_df))
        validation_losses.append(validation_loss/len(valid_df))
    return train_losses, validation_losses


def plot_loss(train_losses, validation_losses):
    plt.plot(train_losses, label='train loss')
    plt.annotate('%0.4f' % train_losses[-1], xy=(1, train_losses[-1]), xytext=(20, 6), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points',
                 arrowprops=dict(arrowstyle= '-')
              )
    plt.plot(validation_losses, label='validation loss')
    plt.annotate('%0.4f' % validation_losses[-1], xy=(1, validation_losses[-1]), xytext=(20, 20), 
                 xycoords=('axes fraction', 'data'), textcoords='offset points',
                 arrowprops=dict(arrowstyle= '-')
              )


    plt.legend()
    plt.figure()
    plt.show()
   

def plot_decision_boundary(network: Network,
                           test_df: pd.DataFrame,
                           h: float = .02):
    
    x_min, x_max = test_df['x'].min(), test_df['x'].max()
    y_min, y_max = test_df['y'].min(), test_df['y'].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    Z = np.zeros(xx.ravel().shape)
    for i, (x, y) in enumerate(zip(xx.ravel(), yy.ravel())):
        z = network.predict(np.array([[x, y]]))
        #print(z.shape)
        if (z.shape[1] == 1):
            Z[i]  = 0 if z < 0.5 else 1
        else:
            Z[i] = np.argmax(z, axis=1)
    #print(Z)
    Z = Z.reshape(xx.shape)
    #print(Z)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    #plt.axis('off')
    plt.scatter(test_df['x'], test_df['y'], c=test_df['cls'], cmap=plt.cm.viridis)
    

