import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# setup device designation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ES1D model
class RecreationES1D(nn.Module):

    def __init__(self, device):

        self.conv_layers = None
        self.inception = None
        self.fc_layers = None
        self.model = None
        self.device = device

    def build_inception(self):
        # Inception module
        # - Inception a: (1x85)
        # - Inception b: (3x85)
        # - Inception c: (5x86)
        # - kernel size: 3 (for all inception modules)
        # - input shape: (605, 128)
        # - output shape: (603, 256)
        # - all inception modules are concatenated and fed back into the network, made separate from the rest of the network for clarity

        inception = nn.Sequential(
            # Inception a
            nn.Conv1d(128, 85, 1),
            nn.ReLU(),
            # Inception b
            nn.Conv1d(128, 85, 3),
            nn.ReLU(),
            # Inception c
            nn.Conv1d(128, 86, 5),
            nn.ReLU(),
        ).to(self.device)

        return inception

    def set_inception(self):
        self.inception = self.build_inception()

    def build_conv_layers(self):
        # Note: for each layer description the (X x Y) notation means
        # - X is the length of the convolutional kernel
        # - Y is the output feature spaces of that layer
        # define the model with the following layers and parameters:
        # Input:
        # - samples per window: 250
        # - channels: 14
        # - 6 windows per segment

        conv_layers = nn.Sequential(

            # 1. Conv layer (7 x 32)
            # - input shape: (1250, 14)
            # - output shape: (1244, 32)
            nn.Conv1d(14, 32, 7),
            nn.ReLU(),

            # Max pooling layer
            # - pooling window size: 2
            # - input shape: (1244, 32)
            # - output shape: (622, 32)
            nn.MaxPool1d(2),

            # 2. Conv layer (5 x 64)
            # - input shape: (622, 32)
            # - output shape: (618, 64)
            nn.Conv1d(32, 64, 5),
            nn.ReLU(),

            # 3a. Conv layer (5 x 128)
            # - input shape: (618, 64)
            # - output shape: (614, 128)
            nn.Conv1d(64, 128, 5),
            nn.ReLU(),

            # 3b. Conv layer (10 x 128)
            # - input shape: (614, 128)
            # - output shape: (605, 128)
            nn.Conv1d(128, 128, 10),
            nn.ReLU(),

            # add the inception module
            self.inception,


            # 4a. Conv layer (3 x 256)
            # - input shape: (603, 256)
            # - output shape: (601, 256)
            nn.Conv1d(128, 256, 3),
            nn.ReLU(),

            # 4b. Conv layer (3 x 128)
            # - input shape: (603, 256)
            # - output shape: (601, 128)
            nn.Conv1d(256, 128, 3),
            nn.ReLU(),

            # 5a. Conv layer (3 x 64)
            # - input shape: (601, 256)
            # - output shape: (599, 64)
            nn.Conv1d(128, 64, 3),
            nn.ReLU(),

            # 5b. Conv layer (3 x 32)
            # - input shape: (599, 64)
            # - output shape: (597, 32)
            nn.Conv1d(64, 32, 3),
            nn.ReLU(),

            # Max pooling layer (2)
            # - pooling window size: 2
            # - input shape: (597, 32)
            # - output shape: (298, 32)
            nn.MaxPool1d(2)
        ).to(device)

        return conv_layers

    def set_conv_layers(self):
        self.conv_layers = self.build_conv_layers()

    def build_fc_layers(self, num_subjects):
        fc_layers = nn.Sequential(
            # dense layer 1
            nn.Linear(298 * 32, 200),
            nn.ReLU(),
            # dense layer 2
            nn.Linear(200, 200),
            nn.ReLU(),
            # dense layer 3
            nn.Linear(200, 400),
            nn.ReLU(),
            # output layer
            nn.Linear(400, num_subjects)
        ).to(device)

        return fc_layers

    def set_fc_layers(self, num_subjects):
        self.fc_layers = self.build_fc_layers(num_subjects)

    def build_model(self, num_subjects):
        self.set_conv_layers()
        self.set_inception()
        self.set_fc_layers(num_subjects)

        model = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            self.fc_layers
        ).to(device)

        return model

    def set_model(self, num_subjects):
        self.model = self.build_model(num_subjects)

    def define_coeff_variation(self, x):
        # define the coefficient of variation
        # - x: input data
        # - x: (batch_size, 1, 250, 14)
        # - return: (batch_size, 1, 250, 14)

        # calculate the mean of each channel
        mean = torch.mean(x, dim=2, keepdim=True)
        # calculate the standard deviation of each channel
        std = torch.std(x, dim=2, keepdim=True)
        # calculate the coefficient of variation
        coeff_var = torch.div(std, mean)

        return coeff_var

    # define the forward pass
    def forward(self, x):
        # - x: input data
        # - x: (batch_size, 1, 250, 14)
        # - return: (batch_size, 1, 250, 14)

        # calculate the coefficient of variation
        coeff_var = self.define_coeff_variation(x)
        # concatenate the coefficient of variation with the input data
        x = torch.cat((x, coeff_var), dim=1)
        # reshape the input data
        x = x.view(x.size(0), 1, 250, 14)
        # pass the input data through the network
        x = self.model(x)

        return x

    def single_training_epoch(self, train_data):
        """- Optimizer: adam
        - Criterion: cross-entropy
        - Note: no learning rate was given for adam
        """
        # set the model to training mode
        self.model.train()

        # define the optimizer
        optimizer = optim.Adam(self.model.parameters())

        # define the criterion
        criterion = nn.CrossEntropyLoss()

        # define the training loss
        train_loss = 0.0

        # define the number of correct predictions
        correct = 0

        # define the number of samples
        num_samples = 0

        # iterate over the training data
        for round in train_data:

            data = round[0].to(device)
            target = round[1].to(device)

            # move the data to the device
            data, target = data.to(self.device), target.to(self.device)

            # reset the gradients
            optimizer.zero_grad()

            # perform the forward pass
            output = self.model(data)

            # calculate the loss
            loss = criterion(output, target)

            # perform the backward pass
            loss.backward()

            # update the weights
            optimizer.step()

            # add the loss to the training loss
            train_loss += loss.item()

            # get the predictions
            pred = output.argmax(dim=1, keepdim=True)

            # add the number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

            # add the number of samples
            num_samples += len(data)

            # calculate the training loss
            train_loss /= len(train_data.dataset)

        # calculate the training accuracy
        train_acc = correct / num_samples

        return train_loss, train_acc

    def single_validation_epoch(self, val_data):
        """
        - Optimizer: adam
        - Criterion: cross-entropy
        - Note: no learning rate was given for adam
        """

        # set the model to evaluation mode
        self.model.eval()

        # define the criterion
        criterion = nn.CrossEntropyLoss()

        # define the validation loss
        val_loss = 0.0

        # define the number of correct predictions
        correct = 0

        # define the number of samples
        num_samples = 0

        # iterate over the validation data
        for round in val_data:

            data = round[0].to(device)
            target = round[1].to(device)

            # move the data to the device
            data, target = data.to(self.device), target.to(self.device)

            # perform the forward pass
            output = self.model(data)

            # calculate the loss
            loss = criterion(output, target)

            # add the loss to the validation loss
            val_loss += loss.item()

            # get the predictions
            pred = output.argmax(dim=1, keepdim=True)

            # add the number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

            # add the number of samples
            num_samples += len(data)

        # calculate the validation loss
        val_loss /= len(val_data.dataset)

        # calculate the validation accuracy
        val_acc = correct / num_samples

        return val_loss, val_acc

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
