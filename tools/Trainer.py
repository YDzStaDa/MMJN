import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
import time

class MNTrainer2(object):
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, model_name, log_dir, device=''):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.device = device

        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)

    def train(self, n_epochs):
        total_epoch_time = 0
        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            for i, data in enumerate(self.train_loader):
                in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()
                self.optimizer.zero_grad()
                out_data = self.model(in_data)
                loss = self.loss_fn(out_data, target)
                self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)
                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())
                acc = correct_points.float() / results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)
                loss.backward()
                self.optimizer.step()
                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch + 1, i + 1, loss, acc)
                if (i + 1) % 30 == 0:
                    print(self.model_name, log_str)
                    with open("Acc.txt", "a") as file:
                        file.write(self.model_name + ", ")
                        file.write(log_str + "\n")
            i_acc += i

            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', loss, epoch + 1)

            # Save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch)

            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            total_epoch_time += epoch_time
            with open("Time.txt", "a") as file:
                file.write(self.model_name + ", " + f"Epoch {epoch + 1} time: {epoch_time} seconds" + "\n")

        avg_epoch_time = total_epoch_time / n_epochs
        print(f"Average epoch time: {avg_epoch_time} seconds")
        with open("Time.txt", "a") as file:
            file.write(self.model_name + ", " + f"Average epoch time: {avg_epoch_time} seconds" + "\n")
        self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0
        wrong_class = np.zeros(10)
        samples_class = np.zeros(10)
        all_loss = 0
        self.model.eval()

        for _, data in enumerate(self.val_loader, 0):
            in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()
            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target
            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print(self.model_name, 'Total # of test models: ', all_points)
        class_acc = (samples_class - wrong_class) / samples_class
        val_mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)
        print(self.model_name, 'val class acc. : ', class_acc)
        print(self.model_name, 'val mean class acc. : ', val_mean_class_acc)
        print(self.model_name, 'val overall acc. : ', val_overall_acc)
        print(self.model_name, 'val loss : ', loss)
        with open("Acc.txt", "a") as file:
            file.write(self.model_name + ", ")
            file.write("val class acc. : " + "\n" + str(class_acc) + "\n")
            file.write("val mean class acc. : " + str(val_mean_class_acc) + "\n")
            file.write("val overall acc. : " + str(val_overall_acc) + "\n")
            file.write("val loss : " + str(loss) + "\n")
        self.model.train()
        return loss, val_overall_acc, val_mean_class_acc