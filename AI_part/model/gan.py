import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

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

# Create DataLoader for batch processing
train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Check if CUDA is available and move model to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Parameters
latent_dim = 100
input_dim = x_train.shape[1]
epochs = 10
lr = 0.0001

# Instantiate Generator and Discriminator
generator = Generator(latent_dim, input_dim).to(device)
discriminator = Discriminator(input_dim).to(device)

# Loss function and optimizers
adversarial_loss = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training loop
for epoch in range(epochs):
    for i, (real_data, _) in enumerate(train_dataloader):
        # Move real data to the appropriate device
        real_data = real_data.to(device)

        # Adversarial ground truths
        valid = torch.ones(real_data.size(0), 1, device=device)
        fake = torch.zeros(real_data.size(0), 1, device=device)

        # -----------------
        # Train Generator
        # -----------------
        g_optimizer.zero_grad()

        # Sample noise as generator input
        z = torch.randn(real_data.size(0), latent_dim, device=device)

        # Generate data
        generated_data = generator(z)

        # Calculate generator loss
        g_loss = adversarial_loss(discriminator(generated_data), valid)

        # Backprop and optimize
        g_loss.backward()
        g_optimizer.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        d_optimizer.zero_grad()

        # Loss for real data
        real_loss = adversarial_loss(discriminator(real_data), valid)

        # Loss for fake data
        fake_loss = adversarial_loss(discriminator(generated_data.detach()), fake)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2

        # Backprop and optimize
        d_loss.backward()
        d_optimizer.step()

        # Print training progress
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch {i}/{len(train_dataloader)} Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

# Save the trained discriminator model
torch.save(discriminator.state_dict(), file_path + 'discriminator_model.pth')
print("Training complete and model saved.")

# Load the trained discriminator model for testing
discriminator.load_state_dict(torch.load(file_path + 'discriminator_model.pth'))
discriminator.eval()

# Create DataLoader for test data
test_dataset = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate the discriminator on the test data
all_test_labels = []
all_test_predictions = []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        # Move inputs to the appropriate device
        inputs = inputs.to(device)

        # Forward pass
        outputs = discriminator(inputs)
        predictions = (outputs.cpu().numpy() > 0.5).astype(int)
        all_test_predictions.extend(predictions)
        all_test_labels.extend(labels.numpy().astype(int))

# Calculate test accuracy and print classification report
test_accuracy = accuracy_score(all_test_labels, all_test_predictions)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(classification_report(all_test_labels, all_test_predictions))

# Supervised training of Discriminator using generated data and real data
supervised_epochs = 3
supervised_d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

for epoch in range(supervised_epochs):
    for i, (real_data, labels) in enumerate(train_dataloader):
        # Move real data to the appropriate device
        real_data, labels = real_data.to(device), labels.to(device)

        # Sample noise as generator input
        z = torch.randn(real_data.size(0), latent_dim, device=device)

        # Generate fake data
        generated_data = generator(z)

        # Create labels for fake data (0) and real data (1)
        fake_labels = torch.zeros(real_data.size(0), 1, device=device)
        real_labels = torch.ones(real_data.size(0), 1, device=device)

        # Concatenate real and fake data
        combined_data = torch.cat((real_data, generated_data), 0)
        combined_labels = torch.cat((real_labels, fake_labels), 0)

        # Train Discriminator in supervised manner
        supervised_d_optimizer.zero_grad()
        outputs = discriminator(combined_data)
        d_loss = adversarial_loss(outputs, combined_labels)

        # Backprop and optimize
        d_loss.backward()
        supervised_d_optimizer.step()

        # Print training progress
        if i % 100 == 0:
            print(f"Supervised Epoch [{epoch+1}/{supervised_epochs}] Batch {i}/{len(train_dataloader)} Loss D: {d_loss.item():.4f}")

# Save the supervised trained discriminator model
torch.save(discriminator.state_dict(), file_path + 'supervised_discriminator_model.pth')
print("Supervised training complete and model saved.")

# Load the supervised trained discriminator model for final testing
discriminator.load_state_dict(torch.load(file_path + 'supervised_discriminator_model.pth'))
discriminator.eval()

# Evaluate the discriminator on the test data after supervised training
all_test_labels = []
all_test_predictions = []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        # Move inputs to the appropriate device
        inputs = inputs.to(device)

        # Forward pass
        outputs = discriminator(inputs)
        predictions = (outputs.cpu().numpy() > 0.5).astype(int)
        all_test_predictions.extend(predictions)
        all_test_labels.extend(labels.numpy().astype(int))

# Calculate test accuracy and print classification report after supervised training
test_accuracy = accuracy_score(all_test_labels, all_test_predictions)
print(f'Test Accuracy after supervised training: {test_accuracy:.4f}')
print(classification_report(all_test_labels, all_test_predictions))
