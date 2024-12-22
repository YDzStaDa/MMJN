import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
from tools.Trainer import MNTrainer2
from tools.ImgDataset import MMJDataset
from models.MMJN import CNNS1, CNNS2, CNNS3

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="mmjn")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=16)
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=5)
parser.add_argument("-train_path", type=str, default="trainpath")
parser.add_argument("-val_path", type=str, default="testpath")
parser.set_defaults(train=False)

def create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda:?' if torch.cuda.is_available() else 'cpu')
    print('Training on:', device)
    pretraining = not args.no_pretraining
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    with open("Acc.txt", "w") as file:
        pass
    with open("Time.txt", "w") as file:
        pass
    log_dir = args.name+'_MMJN'
    create_folder(log_dir)
    cnet_1 = CNNS1(args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name).to(device)
    cnet_2 = CNNS2(args.name, cnet_1, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views - 1).to(device)
    cnet = CNNS3(args.name, cnet_1, cnet_2, nclasses=40, num_views=args.num_views).to(device)
    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_models_train = args.num_models*args.num_views
    train_dataset = MMJDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    val_dataset = MMJDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = MNTrainer2(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mmjn', log_dir, device=device)
    trainer.train()



