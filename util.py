from torch.autograd import Variable
import numpy as np
import os
from torchvision import datasets, models, transforms
import csv
from matplotlib import pyplot as plt
import torch


def ckpt_save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))


def ckpt_load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    # throw exception
    if not ckpt_lst:
        print()
        print('Checkpoint files are not exist')
        print()
        epoch = 0
        return net, optim, epoch
    else:
        ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

        net.load_state_dict(dict_model['net'])
        optim.load_state_dict(dict_model['optim'])
        epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
        #print(epoch)
        return net, optim, epoch


def csv_save(csv_dir, loss, acc, best_acc):
    #if not os.path.exists(csv_dir):
    #    os.makedirs(csv_dir)
    #print(best_acc, type(best_acc))
    with open(csv_dir + '/loss_acc.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(loss)
        writer.writerow(acc)
        writer.writerow(best_acc)


def csv_load(csv_dir):
    print('Load csv file.. (parse tensor) \n')
    with open(csv_dir + '/loss_acc.csv', 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for i, line in enumerate(rdr):
            if i == 0:
                loss = line
            elif i == 1:
                acc = line
            elif i == 2:
                best_acc = line
    loss_ = []
    acc_ = []
    for i in range(0, len(loss)):
        loss_ += [float(loss[i])]
        acc_ += [float(acc[i])]
    train_loss_list = loss_
    train_acc_list = acc_

    best_acc = torch.tensor(float(best_acc[0]))
    #print(best_acc, type(best_acc))
    return train_loss_list, train_acc_list, [best_acc]


def read_csv(path):
    loss, acc, best_acc = csv_load(path)
    #print(best_acc, type(best_acc))
    best_acc = float(best_acc[0])
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    for i in range(0, len(loss)):
        if i % 2 == 0:
            train_loss += [loss[i]]
            train_acc += [acc[i]]
        else:
            val_loss += [loss[i]]
            val_acc += [acc[i]]

    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel('epoch')
    plt.ylabel('accuracy/loss (%)')
    plt.text(len(train_loss)*0.05, best_acc*0.97, 'Best Accuracy: ' + str(best_acc))
    plt.legend(['train_acc', 'val_acc', 'train_loss', 'val_loss'], loc='center right')
    plt.savefig(path + '/train_val_acc.png')
    plt.show()


def transform_data():
    normalize_list = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(1),
            transforms.ToTensor(),
            transforms.Normalize(normalize_list[0],normalize_list[1])
        ]),
        'val': transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(normalize_list[0], normalize_list[1])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(normalize_list[0], normalize_list[1])
        ]),
    }

    data_dir = './datasets_jdc'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']} #,'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ['train', 'val']}#,'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}#,'test']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names, image_datasets


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# this function is used at train mode but not used now
def visualize_model(model, dataloaders, class_names, use_gpu, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)


def visualize_model_TEST(model, dataloaders, class_names, use_gpu, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        #print(inputs.shape)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {} | org: {}'.format(class_names[preds[j]], class_names[labels[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)


