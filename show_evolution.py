import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_train.csv')
columns_names = data.columns

epoch = data[columns_names[0]]
colors = ['r','b','g','c','m','y']

fig, axs = plt.subplots(2,3,figsize=(15, 6))
axs = axs.ravel()
for i,col in enumerate(columns_names[3:9]):
    axs[i].plot(epoch, data[col],color=colors[i],label=col)
    axs[i].set_title(col)
plt.savefig("Loss_evolution.png",dpi=300)

fig2, axs2 = plt.subplots(3,1,figsize=(7, 6))
axs2 = axs2.ravel()
for i,col in enumerate(columns_names[9:]):
    axs2[i].plot(epoch, data[col],color=colors[i],label=col)
    axs2[i].set_title(col)
plt.savefig("accuracy_evolution.png",dpi=300)

