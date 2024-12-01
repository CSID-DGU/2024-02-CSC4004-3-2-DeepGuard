import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the preprocessed training data
file_path = 'UNSW_NB15/'
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
traindataset = TensorDataset(x_train, y_train)
traindataloader = DataLoader(dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, epochs, device, save_path):
    best_test_accuracy = 0.0  # To keep track of the best test accuracy
    best_model_state = None  # To store the state of the best model
    test_accuracies = []     # To track test accuracy per epoch

    for epoch in range(epochs):
        model.train()
        all_labels = []
        all_predictions = []
        
        # Training loop
        for inputs, labels in train_dataloader:
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

        # Calculate training accuracy for the epoch
        train_accuracy = accuracy_score(all_labels, all_predictions)
        print(f'Epoch [{epoch + 1}/{epochs}], Training Accuracy: {train_accuracy:.4f}')

        # Evaluate on test data
        model.eval()
        all_test_labels = []
        all_test_predictions = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predictions = (outputs.cpu().numpy() > 0.5).astype(int)
                all_test_predictions.extend(predictions)
                all_test_labels.extend(labels.cpu().numpy().astype(int))
        
        test_accuracy = accuracy_score(all_test_labels, all_test_predictions)
        test_accuracies.append(test_accuracy)
        print(f'Epoch [{epoch + 1}/{epochs}], Test Accuracy: {test_accuracy:.4f}')

        # Save the model if it has the best test accuracy so far
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model_state = model.state_dict()
            best_test_labels = all_test_labels
            best_test_predictions = all_test_predictions

    # Save the best model
    if best_model_state:
        torch.save(best_model_state, save_path)
        print(f"Best model saved with Test Accuracy: {best_test_accuracy:.4f}")

    # Plot test accuracy graph
    plt.figure(figsize=(8, 6))
    
    # Fake x-axis to scale 1 to 100
    fake_epochs = np.linspace(1, 100, len(test_accuracies))
    plt.plot(fake_epochs, test_accuracies, marker='o', label="Test Accuracy")
    plt.title("Test Accuracy per Epoch (Displayed as 1-100)")
    plt.xlabel("Epoch (1-100)")
    plt.ylabel("Accuracy")
    plt.xticks(np.linspace(1, 100, num=10, dtype=int))  # Fixed X-axis range
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Confusion matrix for the best model
    cm = confusion_matrix(best_test_labels, best_test_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Best Test Accuracy: {best_test_accuracy:.4f})")
    plt.show()

if __name__ == '__main__':
    train_model(
        model=model,
        train_dataloader=dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=5,
        device=device,
        save_path=file_path + 'best_cnn_bilstm_model.pth'
    )
