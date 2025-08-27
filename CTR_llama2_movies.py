import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data = pd.read_csv('./movies_LLM_4reviews_codebook.csv') # !!用这个index，要用dataloader重新做的index，参考之前做csv

print(data)

data_X = data.iloc[:,[0,1,3,4]] # 不要click作为特征
data_y = data.click.values

fields = data_X.max().values + 1 # 模型输入的feature_fields

epoches = 50 # 可以改epoch数量

def train(model):
    for epoch in range(epoches):
        train_loss = []
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        model.train()
        for batch, (x, y) in enumerate(train_loader):
            pred = model(x) #
            loss = criterion(pred, y.float().detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        val_loss = []
        prediction = []
        y_true = []
        with torch.no_grad():
            for batch, (x, y) in enumerate(val_loader):
                pred = model(x)
                loss = criterion(pred, y.float().detach())
                val_loss.append(loss.item())
                prediction.extend(pred.tolist())
                y_true.extend(y.tolist())
        val_auc = roc_auc_score(y_true=y_true, y_score=prediction)
        print ("EPOCH %s train loss : %.5f   validation loss : %.5f   validation auc is %.5f" % (epoch, np.mean(train_loss), np.mean(val_loss), val_auc))
    return train_loss, val_loss, val_auc


from CTRmodel import DCN00

model = DCN00.DeepCrossNet(feature_fields=fields, embed_dim=8, num_layers=3, mlp_dims=(16,16), dropout=0.2)

_ = train(model)