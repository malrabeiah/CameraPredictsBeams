'''
Main script for training and testing a DL model (resnet18) for mmWave beam prediction
Author: Muhammad Alrabeiah
Aug. 2019
'''

import torch as t
import torch.cuda as cuda
import torch.optim as optimizer
import torch.nn as nn
from data_feed import DataFeed
from torch.utils.data import DataLoader
import torchvision.transforms as transf
import numpy as np
import matplotlib.pyplot as plt
from build_net import resnet18_mod
from skimage import io, transform
from scipy import io


############### Image grab #################
def imageGrab(X, ind, save_path):
    X = X.numpy()
    img = X[0, :, :, :]
    img_rgb = np.zeros((224, 224, 3))
    img_rgb[:, :, 0] = (0.209 * img[0, :, :]) + 0.439
    img_rgb[:, :, 1] = (0.154 * img[1, :, :]) + 0.410
    img_rgb[:, :, 2] = (0.195 * img[2, :, :]) + 0.316
    img_rgb = transform.resize(img_rgb, (540, 960))
    io.imshow(img_rgb)
    io.imsave(save_path + '/test' + str(ind) + '.jpg', img_rgb)


############################################

# Hyper-parameters
batch_size = 150
lr = 1e-4
decay = 1e-4
image_grab = False
num_epochs = 15
train_size = [1]

# Data pre-processing:
img_resize = transf.Resize((224, 224))
img_norm = transf.Normalize(mean=(0.306, 0.281, 0.251), std=(0.016, 0.0102, 0.013))
proc_pipe = transf.Compose(
    [transf.ToPILImage(),
     img_resize,
     transf.ToTensor(),
     img_norm]
)
train_dir = 'train_images'
val_dir = 'test_images'
train_loader = DataLoader(DataFeed(train_dir, nat_sort=True, transform=proc_pipe),
                          batch_size=batch_size,
                          shuffle=False)
val_loader = DataLoader(DataFeed(val_dir, nat_sort=True, transform=proc_pipe),
                        batch_size=batch_size,
                        shuffle=False)


# Network training:
with cuda.device(0):
    top_1 = np.zeros( (1,len(train_size)) )
    top_2 = np.zeros( (1,len(train_size)) )
    top_3 = np.zeros( (1,len(train_size)) )
    acc_loss = 0
    itr = []
    for idx, n in enumerate(train_size):
        print('```````````````````````````````````````````````````````')
        print('Training size is {}'.format(n))
        # Build the network:
        net = resnet18_mod(pretrained=True, progress=True, num_classes=64)
        net = net.cuda()
        layers = list(net.children())

        #  Optimization parameters:
        criterion = nn.CrossEntropyLoss()
        opt = optimizer.Adam(net.parameters(), lr=lr, weight_decay=decay)
        LR_sch = optimizer.lr_scheduler.MultiStepLR(opt, [4, 8], gamma=0.1, last_epoch=-1)

        count = 0
        running_loss = []
        running_top1_acc = []
        running_top2_acc = []
        running_top3_acc = []
        allowed_batches = np.floor((n*3500)/batch_size)
        for epoch in range(num_epochs):
            print('Epoch No. ' + str(epoch + 1))
            skipped_batches = 0
            for tr_count, (img, label) in enumerate(train_loader):
                if tr_count <= allowed_batches:
                    net.train()
                    x = img.cuda()
                    opt.zero_grad()
                    label = label.cuda()
                    out = net.forward(x)
                    L = criterion(out, label)
                    L.backward()
                    opt.step()
                    batch_loss = L.item()
                    acc_loss += batch_loss
                    count += 1
                    if np.mod(count, 10) == 0:
                        print('Training-Batch No.' + str(count))
                        running_loss.append(batch_loss)  # running_loss.append()
                        itr.append(count)
                        print('Loss = ' + str(running_loss[-1]))
                else:
                    skipped_batches += 1

            print('Start validation')
            ave_top1_acc = 0
            ave_top2_acc = 0
            ave_top3_acc = 0
            ind_ten = t.as_tensor([0, 1, 2], device='cuda:0')
            for val_count, (imgs, labels) in enumerate(val_loader):
                net.eval()
                x = imgs.cuda()
                # ind_ten.cuda()
                opt.zero_grad()
                labels = labels.cuda()
                out = net.forward(x)
                _, top_1_pred = t.max(out, dim=1)
                sorted_out = t.argsort(out, dim=1, descending=True)
                top_2_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:2])
                top_3_pred = t.index_select(sorted_out, dim=1, index=ind_ten)
                reshaped_labels = labels.reshape((labels.shape[0], 1))
                tiled_2_labels = reshaped_labels.repeat(1, 2)
                tiled_3_labels = reshaped_labels.repeat(1, 3)
                batch_top1_acc = t.sum(top_1_pred == labels, dtype=t.float32)
                batch_top2_acc = t.sum(top_2_pred == tiled_2_labels, dtype=t.float32)
                batch_top3_acc = t.sum(top_3_pred == tiled_3_labels, dtype=t.float32)
                if top_1_pred.cpu().numpy()[0] == labels.cpu().numpy()[0] and image_grab:
                    imageGrab(x.cpu(), labels.cpu().numpy()[0], '/home/malrabei/Desktop')
                ave_top1_acc += batch_top1_acc.item()
                ave_top2_acc += batch_top2_acc.item()
                ave_top3_acc += batch_top3_acc.item()
            running_top1_acc.append(ave_top1_acc / 1500)  # (batch_size * (count_2 + 1)) )
            running_top2_acc.append(ave_top2_acc / 1500)
            running_top3_acc.append(ave_top3_acc / 1500)  # (batch_size * (count_2 + 1)))
            print('Training_size {}--No. of skipped batchess {}'.format(n,skipped_batches))
            print('Average Top-1 accuracy {}'.format( running_top1_acc[-1]))
            print('Average Top-2 accuracy {}'.format( running_top2_acc[-1]))
            print('Average Top-3 accuracy {}'.format( running_top3_acc[-1]))
            LR_sch.step()
        top_1[0,idx] = running_top1_acc[-1]
        top_2[0,idx] = running_top2_acc[-1]
        top_3[0,idx] = running_top3_acc[-1]

net_name = 'Networks/ResNet18_beam_pred'
t.save(net.state_dict(), net_name)
accuracies = np.concatenate((top_1, top_2, top_3), axis=0)
io.savemat('results', {'results': accuracies})
fig3 = plt.figure()
plt.plot(train_size, np.squeeze(top_1), 'g*-', label='Top-1 Accuracy')
plt.plot(train_size, np.squeeze(top_2), 'b*-', label='Top-2 Accuracy')
plt.plot(train_size, np.squeeze(top_3), 'r*-', label='Top-3 Accuracy')
plt.legend()
plt.grid(True)

plt.show()