Original data transform:

```python
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])])
```

Result:

```bash
best validation accuracy is 92.4300 percent
```


Try data aumentation in the hw4
```python
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomAffine((-5, 5), shear=(-10, 10)),
    transforms.Normalize([0.5], [0.5])])
```

Result:
```
100%|██████████| 1875/1875 [12:12<00:00,  2.56it/s]
train Accuracy: 0.82805
100%|██████████| 313/313 [01:00<00:00,  5.20it/s]
valid Accuracy: 0.888
100%|██████████| 1875/1875 [12:03<00:00,  2.59it/s]
train Accuracy: 0.8989166666666667
100%|██████████| 313/313 [01:01<00:00,  5.11it/s]
valid Accuracy: 0.8846
100%|██████████| 1875/1875 [12:03<00:00,  2.59it/s]
train Accuracy: 0.9131666666666667
100%|██████████| 313/313 [00:59<00:00,  5.24it/s]
valid Accuracy: 0.9069
100%|██████████| 1875/1875 [12:03<00:00,  2.59it/s]
train Accuracy: 0.9222666666666667
100%|██████████| 313/313 [01:01<00:00,  5.09it/s]
valid Accuracy: 0.9103
100%|██████████| 1875/1875 [12:01<00:00,  2.60it/s]
train Accuracy: 0.9295333333333333
100%|██████████| 313/313 [01:01<00:00,  5.11it/s]
valid Accuracy: 0.9172
------------------------------------------------------------
best validation accuracy is 91.7200 percent
```

Try other data augmentation: RandomCrop
```python
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.Normalize([0.5], [0.5])])
```

Result:
```bash
100%|██████████| 1875/1875 [51:11<00:00,  1.64s/it]
train Accuracy: 0.8417166666666667
100%|██████████| 313/313 [01:45<00:00,  2.96it/s]
valid Accuracy: 0.8958
100%|██████████| 1875/1875 [50:59<00:00,  1.63s/it]
train Accuracy: 0.9095333333333333
100%|██████████| 313/313 [01:44<00:00,  3.00it/s]
valid Accuracy: 0.9143
100%|██████████| 1875/1875 [50:55<00:00,  1.63s/it]
train Accuracy: 0.92505
100%|██████████| 313/313 [01:44<00:00,  3.00it/s]
valid Accuracy: 0.9146
100%|██████████| 1875/1875 [50:49<00:00,  1.63s/it]
train Accuracy: 0.9347833333333333
100%|██████████| 313/313 [01:44<00:00,  3.00it/s]
valid Accuracy: 0.9211
100%|██████████| 1875/1875 [50:51<00:00,  1.63s/it]
train Accuracy: 0.94485
100%|██████████| 313/313 [01:46<00:00,  2.95it/s]
valid Accuracy: 0.9207
------------------------------------------------------------
best validation accuracy is 92.1100 percent
```

```python
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.Normalize([0.5], [0.5])]
```

Result:
```bash
100%|██████████| 1875/1875 [12:17<00:00,  2.54it/s]
train Accuracy: 0.6445666666666666
100%|██████████| 313/313 [00:59<00:00,  5.22it/s]
valid Accuracy: 0.7217
100%|██████████| 1875/1875 [12:08<00:00,  2.57it/s]
train Accuracy: 0.7878333333333334
100%|██████████| 313/313 [00:59<00:00,  5.25it/s]
valid Accuracy: 0.8099
100%|██████████| 1875/1875 [12:05<00:00,  2.58it/s]
train Accuracy: 0.8276833333333333
100%|██████████| 313/313 [00:59<00:00,  5.25it/s]
valid Accuracy: 0.8305
100%|██████████| 1875/1875 [12:04<00:00,  2.59it/s]
train Accuracy: 0.8485
100%|██████████| 313/313 [01:00<00:00,  5.20it/s]
valid Accuracy: 0.8514
100%|██████████| 1875/1875 [12:04<00:00,  2.59it/s]
train Accuracy: 0.8606333333333334
100%|██████████| 313/313 [00:59<00:00,  5.25it/s]
valid Accuracy: 0.8555
------------------------------------------------------------
best validation accuracy is 85.5500 percent
```


summary
```bash
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]             576
              ReLU-2         [-1, 64, 224, 224]               0
            Conv2d-3         [-1, 64, 224, 224]          36,928
              ReLU-4         [-1, 64, 224, 224]               0
         MaxPool2d-5         [-1, 64, 112, 112]               0
            Conv2d-6        [-1, 128, 112, 112]          73,856
              ReLU-7        [-1, 128, 112, 112]               0
            Conv2d-8        [-1, 128, 112, 112]         147,584
              ReLU-9        [-1, 128, 112, 112]               0
        MaxPool2d-10          [-1, 128, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]         295,168
             ReLU-12          [-1, 256, 56, 56]               0
           Conv2d-13          [-1, 256, 56, 56]         590,080
             ReLU-14          [-1, 256, 56, 56]               0
        MaxPool2d-15          [-1, 256, 28, 28]               0
           Conv2d-16          [-1, 512, 28, 28]       1,180,160
             ReLU-17          [-1, 512, 28, 28]               0
           Conv2d-18          [-1, 512, 28, 28]       2,359,808
             ReLU-19          [-1, 512, 28, 28]               0
        MaxPool2d-20          [-1, 512, 14, 14]               0
           Conv2d-21          [-1, 512, 14, 14]       2,359,808
             ReLU-22          [-1, 512, 14, 14]               0
           Conv2d-23          [-1, 512, 14, 14]       2,359,808
             ReLU-24          [-1, 512, 14, 14]               0
        MaxPool2d-25            [-1, 512, 7, 7]               0
AdaptiveAvgPool2d-26            [-1, 512, 7, 7]               0
           Linear-27                 [-1, 4096]     102,764,544
             ReLU-28                 [-1, 4096]               0
          Dropout-29                 [-1, 4096]               0
           Linear-30                 [-1, 4096]      16,781,312
             ReLU-31                 [-1, 4096]               0
          Dropout-32                 [-1, 4096]               0
           Linear-33                   [-1, 10]          40,970
              VGG-34                   [-1, 10]               0
================================================================
Total params: 128,990,602
Trainable params: 128,990,602
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 198.87
Params size (MB): 492.06
Estimated Total Size (MB): 691.12
----------------------------------------------------------------
```

model
```bash
VGGNet(
  (model): VGG(
    (features): Sequential(
      (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): ReLU(inplace=True)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace=True)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace=True)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace=True)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace=True)
      (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (15): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (16): ReLU(inplace=True)
      (17): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (18): ReLU(inplace=True)
      (19): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (20): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (21): ReLU(inplace=True)
      (22): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (23): ReLU(inplace=True)
      (24): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
    (classifier): Sequential(
      (0): Linear(in_features=25088, out_features=4096, bias=True)
      (1): ReLU(inplace=True)
      (2): Dropout(p=0.5, inplace=False)
      (3): Linear(in_features=4096, out_features=4096, bias=True)
      (4): ReLU(inplace=True)
      (5): Dropout(p=0.5, inplace=False)
      (6): Linear(in_features=4096, out_features=10, bias=True)
    )
  )
)
```