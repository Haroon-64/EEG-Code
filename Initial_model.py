# %%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

#----------------------------------------------------------------------------
# Parameters
NUM_FILES = None
NUM_CLASSES = 6 
IMG_SIZE = (100, 100)  
BATCH_SIZE = 64  
EPOCHS = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#----------------------------------------------------------------------------
class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, transform=None) -> None:
        self.spectrograms = spectrograms
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx) -> tuple:
        spectrogram = self.spectrograms[idx]
        label = self.labels[idx]
        if self.transform:
            spectrogram = Image.fromarray(spectrogram)
            if spectrogram.mode != 'RGB':
                spectrogram = spectrogram.convert('RGB')
            spectrogram = self.transform(spectrogram)
        return spectrogram, label


def read_data(data_folder, num_files=None) -> tuple[list, list, pd.DataFrame, pd.DataFrame]:
    train_spec_folder = os.path.join(data_folder, 'train_spectrograms')
    test_spec_folder = os.path.join(data_folder, 'test_spectrograms')

    def read_npy_folder(folder_path, n_files=None) -> tuple[list, list]:
        arrays = []
        filenames = []
        files_to_read = os.listdir(folder_path)[:n_files] if n_files else os.listdir(folder_path)
        for file in files_to_read:
            if file.endswith('.npy'):
                file_path = os.path.join(folder_path, file)
                array = np.load(file_path)
                arrays.append(array)
                filenames.append(int(file.split('.')[0]))  # Extracting ID from filename
        print(f"Read {len(arrays)} files from {folder_path}.")
        return arrays, filenames

    train_spec, train_ids = read_npy_folder(train_spec_folder, num_files)
    test_spec, test_ids = read_npy_folder(test_spec_folder)

    train_labels = pd.read_csv(os.path.join(data_folder, 'train.csv'))
    test_labels = pd.read_csv(os.path.join(data_folder, 'test.csv'))

    # Filter labels based on matching ID
    train_labels = train_labels[train_labels['spectrogram_id'].isin(train_ids)]
    test_labels = test_labels[test_labels['spectrogram_id'].isin(test_ids)]

    # Limit the number of labels to match the available data
    train_labels = train_labels.head(len(train_spec))
    test_labels = test_labels.head(len(test_spec))

    return train_spec, test_spec, train_labels, test_labels

#------------------------------------------------------------------------------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes) -> None:
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the correct size of the input to the linear layer
        self.pool_output_size = self.calculate_pool_output_size()
        # self.flatten = torch.flatten()
        self.fc1 = nn.Linear(self.pool_output_size, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(128, num_classes)
        self.batchnorm = nn.BatchNorm1d(128) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)  
        x = self.batchnorm(x)  
        x = self.fc2(x)
        return x
    
    def calculate_pool_output_size(self):
        x = torch.randn(1, 3, 100, 100)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Calculate the flattened size
        return x.size(1) * x.size(2) * x.size(3)

#------------------------------------------------------------------------------------------------------------------------

# Read data
train_spec, test_spec, train_labels, test_labels = read_data('data/npy_data/npy_data', num_files=NUM_FILES)

print(f"Train spec shape: {train_spec.shape}, train labels shape: {train_labels.shape}")  
print(f"Test spec shape: {test_spec.shape}, test labels shape: {test_labels.shape}")

Xt, Xv, yt, yv = train_test_split(
    train_spec, train_labels,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
#------------------------------------------------------------------------------------------------------------------------
# preprocessing labels
y_train = yt.iloc[:, 9:]
y_val = yv.iloc[:, 9:]
pd.DataFrame(y_train)

y_train = y_train.apply(pd.to_numeric, errors='coerce')
y_train.fillna(0, inplace=True)
y_train_normalized = y_train.div(y_train.sum(axis=1), axis=0)
weights = y_train.sum(axis=1)  # Calculate weights based on number of voters per row
y_train_normalized = y_train_normalized.mul(weights, axis=0)
y_train_normalized = y_train_normalized.div(y_train_normalized.sum(axis=1), axis=0)
y_train = torch.tensor(y_train_normalized.values, dtype=torch.float32)

# repeat for validation
y_val = y_val.apply(pd.to_numeric, errors='coerce')
y_val.fillna(0, inplace=True)
y_val_normalized = y_val.div(y_val.sum(axis=1), axis=0)
weights = y_val.sum(axis=1)
y_val_normalized = y_val_normalized.mul(weights, axis=0)
y_val_normalized = y_val_normalized.div(y_val_normalized.sum(axis=1), axis=0)
y_val = torch.tensor(y_val_normalized.values, dtype=torch.float32)

if np.allclose(y_train_normalized.sum(axis=1), 1) and np.allclose(y_val_normalized.sum(axis=1), 1): print("correct normalization" ) 

#------------------------------------------------------------------------------------------------------------------------
#initialize model
model = CNNModel(num_classes=NUM_CLASSES)
model.to(device)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),  
    transforms.ToTensor(),       
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

train_dataset = SpectrogramDataset(Xt, y_train, transform=transform)
val_dataset = SpectrogramDataset(Xv, y_val, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(model.parameters(), lr=0.001)


print(f"length of train dataset: {len(train_dataset)}, length of val dataset: {len(val_dataset)}")

#------------------------------------------------------------------------------------------------------------------------
# training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(F.log_softmax(outputs, dim=1), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)
            loss = criterion(F.log_softmax(outputs, dim=1), labels)
            validation_loss += loss.item() * inputs.size(0)

    epoch_val_loss = validation_loss / len(val_loader.dataset)
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {epoch_val_loss:.4f}")
#------------------------------------------------------------------------------------------------------------------------
# evaluate and save model
from torchsummary import summary
summary(model, (3, 100, 100))

torch.save(model, os.getcwd() + "/saved_model")
