import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset
import random, pdb, math, copy
from loss import CrossEntropyLabelSmooth
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from model import *

def entropy_loss(y_pred, y_true):
    """
    Calculates the entropy loss between predicted and true labels
    
    Args:
    - y_pred: PyTorch tensor of predicted labels (shape: [batch_size, num_classes])
    - y_true: PyTorch tensor of true labels (shape: [batch_size, num_classes])
    
    Returns:
    - loss: scalar entropy loss
    """
    # clip predicted probabilities to avoid taking the log of 0
    y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
    
    # calculate entropy loss
    loss = -torch.mean(torch.sum(y_true * torch.log(y_pred), dim=1))
    
    return loss

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return  transforms.Compose([
          transforms.Resize((resize_size, resize_size)),
          transforms.RandomCrop(crop_size),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize
      ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return  transforms.Compose([
          transforms.Resize((resize_size, resize_size)),
          transforms.CenterCrop(crop_size),
          transforms.ToTensor(),
          normalize
      ])

def data_load(args):
    ## prepare data
    dset_loaders = {}
    input_domain = np.load(args.data_root+'CWRU_'+args.s+'.npy', allow_pickle=True)
    input_domain = input_domain.item()
    input_N = input_domain['Normal']
    input_OR = input_domain['OR']
    input_IR = input_domain['IR']
    # print (np.shape(input_IR), np.shape(input_OR), np.shape(input_N))

    input_label_N = np.zeros([np.size(input_N,0),1])
    input_label_OR = np.ones([np.size(input_OR,0),1])
    input_label_IR = np.ones([np.size(input_IR,0),1])+1

    data = np.concatenate((input_N, input_OR, input_IR) , axis=0)
    print(np.shape(data))
    label = np.concatenate((input_label_N, input_label_OR, input_label_IR), axis=0)
    print(np.shape(label))
    # shuffle inputs
    nums = [x for x in range(np.size(data, axis = 0))]
    random.shuffle(nums)
    data = data[nums, :]
    label = label[nums, :]

    data = np.transpose(data, (0, 2, 1))
    label = np.squeeze(label)

    source_data = Variable(torch.from_numpy(data).float(), requires_grad=False)
    source_label= Variable(torch.from_numpy(label).long(), requires_grad=False)
    source_dataset = TensorDataset(source_data, source_label)
    source_loader = DataLoader(source_dataset,batch_size=args.batch_size)
    
    dset_loaders["source"] = source_loader

    ## load target
    input_domain = np.load(args.data_root+'CWRU_'+args.t+'.npy', allow_pickle=True)
    input_domain = input_domain.item()
    input_N = input_domain['Normal']
    input_OR = input_domain['OR']
    input_IR = input_domain['IR']
    # print (np.shape(input_IR), np.shape(input_OR), np.shape(input_N))

    input_label_N = np.zeros([np.size(input_N,0),1])
    input_label_OR = np.ones([np.size(input_OR,0),1])
    input_label_IR = np.ones([np.size(input_IR,0),1])+1

    data = np.concatenate((input_N, input_OR, input_IR) , axis=0)
    print(np.shape(data))
    label = np.concatenate((input_label_N, input_label_OR, input_label_IR), axis=0)
    print(np.shape(label))
    # shuffle inputs
    nums = [x for x in range(np.size(data, axis = 0))]
    random.shuffle(nums)
    data = data[nums, :]
    label = label[nums, :]

    data = np.transpose(data, (0, 2, 1))
    label = np.squeeze(label)

    target_data = Variable(torch.from_numpy(data).float(), requires_grad=False)
    target_label= Variable(torch.from_numpy(label).long(), requires_grad=False)
    target_dataset = TensorDataset(target_data, target_label)
    target_loader = DataLoader(target_dataset,batch_size=args.batch_size)
    
    dset_loaders["target"] = target_loader
    return dset_loaders

def cal_acc(loader, netG,netC,t=0):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.__next__()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netG.forward(
                inputs)  # a^t
            outputs = netC(outputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent


def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    netG = Generator(source='CWRU_'+args.s, target='CWRU_'+args.t).cuda()
    netC = Classifier(source='CWRU_'+args.s, target='CWRU_'+args.t).cuda()
    
    param_group = []
    learning_rate = args.lr
    for k, v in netG.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.01}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders['source'])
    interval_iter = max_iter // 10
    iter_num = 0


    netG.train()
    netC.train()

    smax=100
    #while iter_num < max_iter:
    for epoch in range(args.max_epoch):
        iter_source = iter(dset_loaders['source'])
        for batch_idx, (inputs_source,
                        labels_source) in enumerate(iter_source):
            # print(inputs_source.shape)
            if inputs_source.size(0) == 1:
                continue

            iter_num += 1
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            feature_src = (netG(inputs_source))

            outputs_source = netC(feature_src)
            classifier_loss = CrossEntropyLabelSmooth(
                num_classes=args.class_num, epsilon=args.smooth)(
                    outputs_source, labels_source)
            # classifier_loss = entropy_loss(outputs_source,labels_source)

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

        #if iter_num % interval_iter == 0 or iter_num == max_iter:
        
        netG.eval()
        netC.eval()
        if args.dset=='visda-2017':
            acc_s_te, acc_list = cal_acc(dset_loaders['source'], netG, netC)
            log_str = 'Task: , Iter:{}/{}; Accuracy = {:.2f}%'.format( iter_num, max_iter, acc_s_te) + '\n' + str(acc_list)
        # args.out_file.write(log_str + '\n')
        # args.out_file.flush()
        print(log_str+'\n')

        if acc_s_te >= acc_init:
            acc_init = acc_s_te
            
            best_netG = netG.state_dict()
            best_netC = netC.state_dict()

      
        netG.train()
        netC.train()

    netG.eval()
    netC.eval()
    
    # acc_s_te, acc_list = cal_acc(dset_loaders['test'], netC, netG,flag= True)

    # log_str = 'Task: {}; Accuracy on target = {:.2f}%'.format(args.name_src, acc_s_te) + '\n' + acc_list
    # args.out_file.write(log_str + '\n')
    # args.out_file.flush()
    # print(log_str+'\n')

    torch.save(best_netG, osp.join(args.output_dir_src, "source_G.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return  netG, netC

def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netG = Generator(source='CWRU_'+args.s, target='CWRU_'+args.t).cuda()
    netC = Classifier(source='CWRU_'+args.s, target='CWRU_'+args.t).cuda()
    
    args.modelpath = args.output_dir_src + '/source_G.pt'
    netG.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netG.eval()
    netC.eval()

    acc, acc_list = cal_acc(dset_loaders['target'],
                                netG,
                                netC
                                )
    log_str = '\nTraining: {}, Task: , Accuracy = {:.2f}%'.format(args.trte,  acc) + '\n' + str(acc_list)

    # args.out_file.write(log_str)
    # args.out_file.flush()
    print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neighbors')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument("--data_root",type= str, default='./CWRU_dataset/')
    parser.add_argument('--s', type=str, default='DE', help="source")
    parser.add_argument('--t', type=str, default='FE', help="target")
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument(
        '--dset',
        type=str,
        default='visda-2017')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='weight/source/')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--class_num',type=int, default=3)
    parser.add_argument('--output_dir_src',default='./model/')
    parser.add_argument('--train',type = str, default='train', choices=['train','test'])
    args = parser.parse_args()

    # if args.dset == 'office-home':
    #     names = ['Art', 'Clipart', 'Product', 'RealWorld']
    #     args.class_num = 65
    # if args.dset == 'visda-2017':
    #     names = ['train', 'validation']
    #     args.class_num = 12


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    folder = './data/'
    args.s_dset_path = args.data_root+'CWRU_'+args.s+'.npy'
    args.train = 'test'
    args.test_dset_path = args.data_root+'CWRU_'+args.t+'.npy'


    # args.output_dir_src = osp.join(args.output, args.da, args.dset, args.s.upper())
    # args.name_src = args.s.upper()
    # if not osp.exists(args.output_dir_src):
    #     os.system('mkdir -p ' + args.output_dir_src)
    # if not osp.exists(args.output_dir_src):
    #     os.mkdir(args.output_dir_src)

    # args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    # args.out_file.write(print_args(args)+'\n')
    # args.out_file.flush()
    train_source(args)
    # args.name = names[args.s][0].upper() + names[args.t][0].upper()
    test_target(args)
