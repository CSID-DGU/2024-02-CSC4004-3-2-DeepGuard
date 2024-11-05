import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# Load the preprocessed training data
file_path = 'AI_part/UNSW_NB15/'
x_train, y_train = pickle.load(open(file_path + 'final_train.pkl', 'rb'))
x_test, y_test = pickle.load(open(file_path + 'final_test.pkl', 'rb'))

# Convert data to NumPy arrays for PyTorch compatibility
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Convert data to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Reshape the data for Conv1D layer
# Conv1D expects input of shape (samples, channels, sequence length)
x_train = x_train.unsqueeze(1)  # Adding channel dimension
x_test = x_test.unsqueeze(1)  # Adding channel dimension

# Create DataLoader for batch processing
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the CNN-BiLSTM model
class CNNBiLSTM(nn.Module):
    def __init__(self):
        super(CNNBiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 2, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # Reshape for LSTM (batch, seq_len, features)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output of the last time step
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Check if CUDA is available and move model to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNBiLSTM().to(device)

# Convert data to the appropriate device
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

# Create DataLoader for batch processing (updated to use CUDA tensors)
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the loss function and the optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
'''
# Training loop
epochs = 2
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    for i, (inputs, labels) in enumerate(dataloader):
        # Move inputs and labels to the appropriate device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        labels = labels.unsqueeze(1)  # Reshape labels for BCELoss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Collect predictions and true labels for accuracy calculation
        predictions = (outputs.detach().cpu().numpy() > 0.5).astype(int)
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy().astype(int))

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    # Calculate accuracy for the epoch
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Epoch [{epoch + 1}/{epochs}], Accuracy: {accuracy:.4f}')

# Save the trained model
torch.save(model.state_dict(), file_path + 'cnn_bilstm_model.pth')

print("Model training complete and saved as cnn_bilstm_model.pth")
'''

model.load_state_dict(torch.load(file_path + 'cnn_bilstm_model.pth'))
model.eval()
# Evaluate the model on the test data
x_test = x_test.to(device)
y_test = y_test.to(device)
test_dataset = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
all_test_labels = []
all_test_predictions = []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        # Move inputs and labels to the appropriate device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        predictions = (outputs.cpu().numpy() > 0.5).astype(int)
        all_test_predictions.extend(predictions)
        all_test_labels.extend(labels.cpu().numpy().astype(int))

# Calculate test accuracy
test_accuracy = accuracy_score(all_test_labels, all_test_predictions)
print(f'Test Accuracy: {test_accuracy:.4f}')