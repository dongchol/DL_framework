from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from util import * #transform_data, visualize_model_TEST, csv_save, csv_load, read_csv, ckpt_save, ckpt_load
from torch.optim import lr_scheduler
from model import Model
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import copy
import torch
import csv

dataloaders, dataset_sizes, class_names, image_datasets = transform_data()
use_gpu = torch.cuda.is_available()

def run(args):
    mode = args.mode
    train_continue = args.train_continue
    lr = args.lr
    model = args.model
    log_dir = args.log_dir
    ckpt_dir = args.ckpt_dir
    csv_dir = args.csv_dir
    num_epoch = args.num_epoch

    if mode == 'train':
        model_finetuning = Model(model, 2)
        model_ft = model_finetuning.initialize_model()
        print(model_ft)

        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=lr)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        # need to declare dataset_size used at Train class_train model
        m = Train(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                  num_epochs=num_epoch, train_continue=train_continue, ckpt_dir=ckpt_dir,
                  csv_dir=csv_dir, log_dir=log_dir)

        m.train_model()
        read_csv(csv_dir)
        # visualize_model(ms, dataloaders, class_names, use_gpu)

    elif mode == 'test':
        net = Model(args.model, 2)
        model_ft = net.initialize_model()
        optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=lr)
        model, optim, st_epoch = ckpt_load(ckpt_dir=ckpt_dir, net=model_ft, optim=optimizer_ft)
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        m = TEST(model)

        ms = m.test_model()
        visualize_model_TEST(ms, dataloaders, class_names, use_gpu)

    plt.ioff()
    plt.show()


class Train:
    def __init__(self, model, criterion, optimizer, scheduler, num_epochs, train_continue,
                 ckpt_dir, csv_dir, log_dir):
        self.model = model  # model
        self.criterion = criterion  # loss
        self.num_epochs = num_epochs  # epoch
        self.optimizer = optimizer  # optimizer
        self.scheduler = scheduler  # learning_rate scheduler
        self.train_continue = train_continue
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.csv_dir = csv_dir
        self.use_gpu = torch.cuda.is_available()
        self.writer_train_loss = SummaryWriter(log_dir=self.log_dir+'/train_loss')
        self.writer_train_acc = SummaryWriter(log_dir=self.log_dir+'/train_acc')

    def train_model(self):  # , model, criterion, optimizer, scheduler, num_epochs=24):
        since = time.time()
        train_loss_list = []
        train_acc_list = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        st_epoch = 0
        if self.train_continue == "on":
            model, optim, st_epoch = ckpt_load(ckpt_dir=self.ckpt_dir, net=self.model, optim=self.optimizer)
            if st_epoch != 0:
                self.model = model
                self.optimizer = optim

                train_loss_list, train_acc_list, best_acc = csv_load(self.csv_dir)
                if str(type(best_acc)) == "<class 'list'>":
                    best_acc = best_acc[0]

        for epoch in range(st_epoch+1, self.num_epochs + 1):
            print('{:.0f}m {:.0f}s            Epoch | {}/{}'.format((time.time() - since) // 60,
                                                                    (time.time() - since) % 60, epoch, self.num_epochs))
            print('-' * 30)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.scheduler.step()
                    self.model.train(True)  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data in dataloaders[phase]:
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    if self.use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    # forward
                    outputs = self.model(inputs)
                    # print('out', outputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = self.criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    # statistics
                    running_loss += loss.data * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                train_loss_list += [epoch_loss.cpu().numpy()]
                train_acc_list += [epoch_acc.cpu().numpy()]
                #TODO: check the parameter of save function, need to change
                if epoch % 2 == 0 or epoch == self.num_epochs:
                    ckpt_save(ckpt_dir=self.ckpt_dir, net=self.model, optim=self.optimizer, epoch=epoch)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            self.writer_train_loss.add_scalar('loss_train', epoch_loss, epoch)
            self.writer_train_acc.add_scalar('acc_train', epoch_acc, epoch)
            print('              save scalar ... ')
            #if str(type(best_acc)) == "<class 'numpy.ndarray'>":
            #    csv_save(self.csv_dir, train_loss_list, train_acc_list, best_acc)
            #else:
            csv_save(self.csv_dir, train_loss_list, train_acc_list, [best_acc.cpu().numpy()])

        # end of for
        if st_epoch == self.num_epochs:
            print('\nCheckpoint: %d epoch already exist\n' %(self.num_epochs))
            print()
        else:
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}\n\n'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model


class TEST:
    def __init__(self, model, num_epochs=1):
        self.model = model  # model
        self.num_epochs = num_epochs
        self.use_gpu = torch.cuda.is_available()

    def test_model(self):  # model, criterion, optimizer, scheduler
        print('  |:    Test Model   :|  ')
        since = time.time()

        for epoch in range(self.num_epochs):
            print(
                '{:.0f}m {:.0f}s                 TEST '.format((time.time() - since) // 60, (time.time() - since) % 60))
            print('-' * 30)

            # Each epoch has a training and validation phase
            for phase in ['val']:
                if phase == 'val':
                    self.model.eval()

                running_corrects = 0

                # Iterate over data.
                for data in dataloaders[phase]:
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    if self.use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # forward
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs.data, 1)

                    # statistics_acc
                    running_corrects += torch.sum(preds == labels.data)

                    epoch_acc = running_corrects / dataset_sizes[phase]
                print('Acc: {:.4f}'.format(epoch_acc))

        time_elapsed = time.time() - since
        print('TEST complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('-' * 30)
        print()
        print()
        return self.model
