import matplotlib.pyplot as plt
import numpy as np


x = [i for i in range(1, 6)]


'''
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
'''
train_acc = [0.82805, 0.8989166666666667, 0.9131666666666667, 0.9222666666666667, 0.9295333333333333]
test_acc = [0.888, 0.8846, 0.9069, 0.9103, 0.9172]

fig, ax = plt.subplots()
ax.plot(x, train_acc, label='train_acc')
ax.plot(x, test_acc, label='valid_acc')
ax.set_title('VGG Try #1')
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_xticks(range(1, 6))
ax.legend()


'''
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
'''
train_acc = [0.8417166666666667, 0.9095333333333333, 0.92505, 0.9347833333333333, 0.94485]
test_acc = [0.8958, 0.9143,  0.9146, 0.9211,  0.9207]

fig, ax = plt.subplots()
ax.plot(x, train_acc, label='train_acc')
ax.plot(x, test_acc, label='valid_acc')
ax.set_title('VGG Try #2')
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_xticks(range(1, 6))
ax.legend()


'''
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
'''
train_acc = [0.6445666666666666, 0.7878333333333334, 0.8276833333333333, 0.8485, 0.8606333333333334]
test_acc = [0.7217, 0.8099, 0.8305, 0.8514, 0.8555]

fig, ax = plt.subplots()
ax.plot(x, train_acc, label='train_acc')
ax.plot(x, test_acc, label='valid_acc')
ax.set_title('VGG Try #3')
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_xticks(range(1, 6))
ax.legend()


train_loss = [0.41793229742646215, 0.23858909827222427, 0.19162874861483772, 0.1552065553434814, 0.12001222678354631]
valid_loss = [0.27857588680264667, 0.24428160998006218, 0.24746710313751866, 0.24466085340827703, 0.22671007265310034]
train_acc = [0.8473666666666667, 0.91445, 0.93035, 0.9434333333333333, 0.9562]
valid_acc = [0.8974, 0.9097, 0.9158, 0.9185, 0.9236]
fig, ax = plt.subplots()
ax.plot(x, train_acc, label='train_acc')
ax.plot(x, valid_acc, label='valid_acc')
ax.plot(x, train_loss, label='train_loss')
ax.plot(x, valid_loss, label='valid_loss')
ax.set_title('VGG Baseline')
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy and loss')
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_xticks(range(1, 6))
ax.legend()


plt.show()
