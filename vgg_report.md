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