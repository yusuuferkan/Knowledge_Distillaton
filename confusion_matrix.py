#gerekli kütüphanleri importlama
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import keras.utils
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim

tf.random.set_seed(3)

# Veri kümesini yükleme
columns = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
train_data = pd.read_csv("adult.data", names=columns, na_values=" ?", skipinitialspace=True)
test_data = pd.read_csv("adult.test", names=columns, na_values=" ?", skipinitialspace=True, skiprows=1)

# Eğitim ve test verileri
x_train = train_data.drop("income", axis=1)
y_train = train_data["income"]
x_test = test_data.drop("income", axis=1)
y_test = test_data["income"]

train_data['income'] = train_data['income'].replace({'<=50K': 0, '>50K': 1})
#train_data
test_data['income'] = test_data['income'].replace({'<=50K.': 0, '>50K.': 1})
#test_data

# One-Hot Encoding
def apply_one_hot_encoding(data, categorical_columns):
    encoder = OneHotEncoder()
    encoded_data = pd.get_dummies(data, columns=categorical_columns)
    return encoded_data

categorical_columns = ["workclass", "education", "marital-status", "occupation",
                       "relationship", "race", "sex", "native-country"]

x_train_encoded = apply_one_hot_encoding(x_train, categorical_columns)
x_test_encoded = apply_one_hot_encoding(x_test, categorical_columns)

y_train_encoded = apply_one_hot_encoding(y_train, categorical_columns)
y_test_encoded = apply_one_hot_encoding(y_test, categorical_columns)

# True False'larır 0 ve 1'ler dönüştürme
y_train = y_train.replace({'<=50K': 0, '>50K': 1})
y_test = y_test.replace({'<=50K': 0, '>50K': 1})

#float yapma
x_train_encoded.values
x_test_encoded.values
x_train_encoded = x_train_encoded.astype(float)
x_test_encoded = x_test_encoded.astype(float)

y_train_encoded.values
y_test_encoded.values
y_train_encoded = y_train_encoded.astype(float)
y_test_encoded = y_test_encoded.astype(float)

# tensorlara dönüştürme
x_train_tensor = torch.tensor(x_train_encoded.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_encoded.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

x_train_tensor = ((x_train_tensor)-(x_train_tensor.min(axis=0)[0]))/((x_train_tensor.max(axis=0)[0])-(x_train_tensor.min(axis=0)[0]))
x_test_tensor = ((x_test_tensor)-(x_test_tensor.min(axis=0)[0]))/((x_test_tensor.max(axis=0)[0])-(x_test_tensor.min(axis=0)[0]))

y_train_tensor = ((y_train_tensor)-(y_train_tensor.min(axis=0)[0]))/((y_train_tensor.max(axis=0)[0])-(y_train_tensor.min(axis=0)[0]))
y_test_tensor = ((y_test_tensor)-(y_test_tensor.min(axis=0)[0]))/((y_test_tensor.max(axis=0)[0])-(y_test_tensor.min(axis=0)[0]))

batch_size = 64
feature_num = 108

x = torch.randn(batch_size, feature_num)

x_train_tensor, x_val_tensor, y_train_tensor, y_val_tensor = train_test_split(x_train_tensor, y_train_tensor, test_size=0.2, random_state=3)

print(x_train_tensor.shape)
print(x_val_tensor.shape)
print(y_train_tensor.shape)
print(y_val_tensor.shape)


batch_size = 64
feature_num = 108

x = torch.randn(batch_size, feature_num)

# RecursiveMLPBlock modeli
class RecursiveMLPBlock(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_units=[], 
                 output_dim=None,
                 output_activation=None
                 ):
        super().__init__()
        
        all_units = [input_dim] + hidden_units
        hidden_activations = nn.ReLU()
        if len(all_units) > 1:
            dense_layers = []
            dense_layers.append(nn.Linear(all_units[-2], all_units[-1]))
            dense_layers.append(hidden_activations)
            if output_dim is not None:
                self.output_lin = nn.Linear(hidden_units[-1], output_dim)
            else:
                self.output_lin = nn.Identity()
            if output_activation is not None:
                self.output_act = nn.Sigmoid()
            else:
                self.output_act = nn.Identity()
            self.mlp = nn.Sequential(*dense_layers)
            if len(all_units) > 2:
                self.sub_mlp = RecursiveMLPBlock(input_dim, hidden_units[:-1], None, None)
            else:
                self.sub_mlp = None
        else:
            self.mlp = None
            self.sub_mlp = None
    
    def forward(self, inputs):
        if self.mlp:
            if self.sub_mlp:
                prev_output = self.sub_mlp(inputs)
                output = self.mlp(prev_output) + prev_output
                return self.output_act(self.output_lin(output))
            else:
                output = self.mlp(inputs)
                return self.output_act(self.output_lin(output))

y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)

recursive_model = RecursiveMLPBlock(feature_num, [100, 100, 100],
                                    output_activation = "sigmoid", output_dim = 1)

# optimizer ve loss fonksiyonu tnaımla
sigmoid = nn.Sigmoid()
criterion_kd = nn.BCELoss()  # binary
optimizer_recursive = torch.optim.Adam(recursive_model.parameters(), lr=0.0001)
temperature = 10.0

class DataLoader:
    def __init__(self, x_train, y_train, batch_size=64):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.x_train) / self.batch_size)

    def __iter__(self):
        for i in range(self.batch_size):
            start = i * self.batch_size
            end = min(len(self.x_train), (i + 1) * self.batch_size)
            x_batch = self.x_train[start:end]
            y_batch = self.y_train[start:end]
            yield x_batch, y_batch

# Data loader
data_loader = DataLoader(x_train_tensor, y_train_tensor, batch_size=64)

# Modeli eğitme
recursive_model.train()
epochs = 100

for epoch in range(epochs):
    for x_batch, y_batch in data_loader:
        outputs = recursive_model(x_batch)
        loss = criterion_kd(outputs, y_batch.view(-1, 1))
        loss.backward()
        optimizer_recursive.step()
        optimizer_recursive.zero_grad()

#print(recursive_model(x))
#print(f"Epoch: {epoch + 1}, loss: {loss:.4f}, accuracy: {accuracy:.4f}")  

recursive_model.eval()

def calculate_confusion_matrix(y_true, y_pred):

  confusion_matrix = np.zeros((2, 2))
  for i in range(len(y_true)):
    true_label = int(y_true[i])
    pred_label = int(y_pred[i])
    confusion_matrix[true_label, pred_label] += 1

  return confusion_matrix

with torch.no_grad():
    recursive_predictions = (recursive_model(x_train_tensor) >= 0.5).float()
    recursive_confusion_matrix = calculate_confusion_matrix(y_train_tensor, recursive_predictions)

print("RecursiveMLPBlock Model Confusion Matrix:")
print(recursive_confusion_matrix)

plt.figure(figsize=(10, 10))
plt.imshow(recursive_confusion_matrix, cmap="Blues")
plt.title("RecursiveMLPBlock Model Confusion Matrix")
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.show()

loss_kdn_fn = nn.BCELoss()

epochs = 10
for epoch in range(epochs):
    train_loss_kdn = 0
    train_acc = 0
    
    for i in range(0, len(x_train_tensor), batch_size):
        x_batch = x_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]
        
        y_pred = recursive_model(x_batch)
        
        loss_kdn = loss_kdn_fn(y_pred, y_batch.view([-1 ,1]))
        
        loss_kdn.backward()
        optimizer_recursive.step()
        optimizer_recursive.zero_grad()
        
        train_loss_kdn += loss_kdn.item()
        train_acc += torch.sum(y_pred.round() == y_batch).item()
    
    train_loss_kdn = train_loss_kdn / len(x_train_tensor)
    train_acc = train_acc / len(x_train_tensor)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss_kdn:.4f}")
    print(train_acc)
"""
"""

batch_size = 16
feature_num = 108

x = torch.randn(batch_size, feature_num)

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(feature_num, 1)
            )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(feature_num, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
            )
        
        self.mlp3 = nn.Sequential(
            nn.Linear(feature_num, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
            )
        
        self.mlp4 = nn.Sequential(
            nn.Linear(feature_num, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
            )
        
        self.out_act = nn.Sigmoid()
        
    def forward(self, x):
        output = self.mlp1(x) + self.mlp2(x) + self.mlp3(x) + self.mlp4(x)
        return self.out_act(output)
    

# Öğretmen modeli tanımlama
input_dim = len(x_train_encoded.columns)
baseline_model = Baseline()

# Optimizer ve loss fonksiyonunu tanımlama
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.01)
temperature = 10.0

# Modeli eğitme (Baseline)
epochs = 100
for epoch in range(epochs):
    outputs = baseline_model(x_train_tensor)
    loss_bs = criterion(outputs, y_train_tensor.view(-1, 1))
    loss_bs.backward()
    optimizer.step()
    optimizer.zero_grad()

# Modeli eğitme (recursive)
epochs_kd = 100
for epoch in range(epochs_kd):
    outputs_teacher = baseline_model(x_train_tensor)
    outputs_student = recursive_model(x_train_tensor)
    optimizer_recursive.zero_grad()
    loss_kd = criterion_kd(outputs_student, outputs_teacher)
    
    loss_kd.backward()
    optimizer_recursive.step()

baseline_model(x)
recursive_model(x)

print(baseline_model(x))
print("\n\n\n")
print(recursive_model(x))

# Modelleri değerlendirme
with torch.no_grad():
    baseline_predictions = (baseline_model(x_train_tensor) >= 0.5).float()
    baseline_accuracy = accuracy_score(y_train_tensor, baseline_predictions)
    
    recursive_predictions = (recursive_model(x_train_tensor) >= 0.5).float()
    recursive_accuracy = accuracy_score(y_train_tensor, recursive_predictions)

print("RecursiveMLPBlock Model Confusion Matrix: \n", recursive_confusion_matrix)
print("\nRecursiveMLPBlock eğitilmeden önce Model Loss: ",loss)
print("Baseline Model Loss: ",loss_bs)
print("RecursiveMLPBlock eğitildikten sonra Model Loss: ",loss_kd)

print("\nBaseline Model Accuracy: ", baseline_accuracy)
print("RecursiveMLPBlock After Baselin Accuracy: ", recursive_accuracy)









