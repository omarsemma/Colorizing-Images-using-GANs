import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

# Importing Files
train_data = pd.read_csv('data_train.csv')
validation_data = pd.read_csv('data_validation.csv')
columns_names = [x for x in train_data.columns][3:]

# Preparing Dataframes of our new Data
dict_train=dict()
dict_validation = dict()
for col in columns_names:
    dict_train[col] = []
    dict_validation[col] = []
    for j in range(0,train_data.size+1,94):
        dict_train[col].append(train_data[col][j+1:94+j].describe()['mean'])
    for k in range(0,validation_data.size+1,63):
        dict_validation[col].append(validation_data[col][k+1:63+k].describe()['mean'])

# Creating Dataframes of our new Data
train_metrics = pd.DataFrame(data=dict_train)
train_metrics = train_metrics.dropna()
validation_metrics = pd.DataFrame(data=dict_validation)
validation_metrics = validation_metrics.dropna()

# Showing plots
def show_fig(epoch,metrics):
    # Metrics = 0 : Show Losses 
    epoch_ax = [x for x in range(epoch+1)]
    fig_size = (2,3) if metrics==0 else (1,3)
    fig, axs = plt.subplots(fig_size[0],fig_size[1],figsize=(15, 6)) 
    tit = "Model Losses" if metrics==0 else "Model Accuracy"
    filename = "Model-losses.png" if metrics==0 else "Model-accuracy.png"
    fig.suptitle(tit+" | Number of Epochs = {}".format(epoch), fontsize=14)
    axs = axs.ravel()
    columns = train_metrics.columns[:6] if metrics==0 else train_metrics.columns[6:]
    for i,col in enumerate(columns):
        axs[i].plot(epoch_ax, train_metrics[col].values,label="Train {}".format(col))
        axs[i].plot(epoch_ax[:epoch], validation_metrics[col].values,label="Validation {}".format(col))
        axs[i].set_title(col)
        axs[i].legend(loc="upper right")
    plt.savefig(filename,dpi=300)
show_fig(149,0)
show_fig(149,1)