# %%
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data.dataloader import default_collate
from torchsummary import summary


# %%
# Reference: https://pytorch.org/hub/pytorch_vision_vgg/
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        # Take vgg13 untrained skeleton
        self.model = models.vgg13(weights=None)
        # Since original vgg13 has a 3-channel input, we have to change it to 1-channel for greyscale
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Similarly, change final output nodes to 10, according to fashion-mnist's class
        self.model.classifier[6] = nn.Linear(4096, 10)

    def forward(self, x):
        return self.model(x)


# %%
model = VGGNet()
if torch.cuda.is_available():
    model.cuda()


# %%
# print model architecture
model


# %%
# print model summary
summary(model, (1, 224, 224))


# %%
# Get data loader Function referenced from hw4, loader.py
# Download Fashion MNIST dataset, apply transforms, fit data into dataloader
def get_data_loader(train_transformer, valid_transformer, batch_size):
    train_loader = DataLoader(torchvision.datasets.FashionMNIST(
        download=True, root=".", transform=train_transformer, train=True), batch_size=batch_size, shuffle=True, pin_memory=True)

    val_loader = DataLoader(torchvision.datasets.FashionMNIST(download=False, root=".",
                            transform=valid_transformer, train=False), batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader


# %%
epochs = 5
batch_size = 32

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.Normalize([0.5], [0.5])])
train_loader, val_loader = get_data_loader(data_transform, data_transform, batch_size)


# %%
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
train_losses = []
valid_losses = []
train_accs = []
valid_accs = []


# %%
# Run function referenced from hw4, helper.py
# Perform forward propagation
# If mode is training, also perform backward propagation and optimize parameters
def run(mode, dataloader, model, optimizer=None, use_cuda=torch.cuda.is_available(), device=None):
    """
    mode: either "train" or "valid". If the mode is train, we will optimize the model
    """
    running_loss = []
    criterion = nn.CrossEntropyLoss()

    actual_labels = []
    predictions = []
    for inputs, labels in tqdm.tqdm(dataloader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss.append(loss.item())

        actual_labels += labels.view(-1).cpu().numpy().tolist()
        _, pred = torch.max(outputs, dim=1)

        predictions += pred.view(-1).cpu().numpy().tolist()

        if mode == "train":
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    acc = np.sum(np.array(actual_labels) == np.array(
        predictions)) / len(actual_labels)
    print(mode, "Accuracy:", acc)

    loss = np.mean(running_loss)

    return loss, acc


# %%
# Perform the actual training and validating process for each epoch
for epoch in range(epochs):
    loss, acc = run("train", train_loader, model, optimizer)
    train_losses.append(loss)
    train_accs.append(acc)
    with torch.no_grad():
        loss, acc = run("valid", val_loader, model, optimizer)
        valid_losses.append(loss)
        valid_accs.append(acc)

print("-"*60)
print("best validation accuracy is %.4f percent" % (np.max(valid_accs) * 100))

# save the model for future reference
torch.save(model, "%s.pt" % str(valid_accs[-1]))
