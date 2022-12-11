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

# PyTorch ResNet-18 is used to FashionMnist dataset
# Reference: https://colab.research.google.com/github/kjamithash/Pytorch_DeepLearning_Experiments/blob/master/FashionMNIST_ResNet_TransferLearning.ipynb
class MnistResNet(nn.Module):
    def __init__(self):
        super(MnistResNet, self).__init__()
        # Take resnet18 untrained skeleton
        self.model = models.resnet18(pretrained=False)
        # Since original resnet18 has a 3-channel input, we have to change it to 1-channel for greyscale
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Similarly, change final output nodes to 10, according to fashion-mnist's class
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
    def forward(self, x):
        return self.model(x)

model = MnistResNet()
if torch.cuda.is_available():
    model = model.cuda()

# Get data loader Function referenced from hw4, loader.py
# Download Fashion MNIST dataset, apply transforms, fit data into dataloader
def get_data_loader(train_transformer, valid_transformer, batch_size):
    train_loader = DataLoader(torchvision.datasets.FashionMNIST(download=True, root=".", transform=train_transformer, train=True),
                              batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(torchvision.datasets.FashionMNIST(download=False, root=".", transform=valid_transformer, train=False),
                            batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

epochs = 5
batch_size = 180
data_transform = transforms.Compose([ transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     #transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.Normalize([0.5], [0.5])])
train_loader, val_loader = get_data_loader(data_transform, data_transform, batch_size)
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
train_losses = []
valid_losses = []
train_accs = []
valid_accs = []


# Run function referenced from hw4, helper.py
# Perform forward propagation
# If mode is training, also perform backward propagation and optimize parameters
def run(mode, dataloader, model, optimizer=None, use_cuda = torch.cuda.is_available()):
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
print("best validation accuracy is %.4f percent" % (np.max(valid_accs) * 100) )

# plot accuracy and loss versus epoch
plt.plot(range(epochs), train_losses, label = 'train_loss')
plt.plot(range(epochs), train_accs, label = 'train_acc')
plt.plot(range(epochs), valid_losses, label = 'valid_loss')
plt.plot(range(epochs), valid_accs, label = 'valid_acc')
plt.legend()
plt.savefig("%s.png" % str(valid_accs[-1]))
plt.show()

# save the model for future reference
torch.save(model, "%s.pt" % str(valid_accs[-1]))

