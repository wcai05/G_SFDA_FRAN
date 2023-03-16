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
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
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
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


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
    indices = torch.arange(len(source_dataset))
    index_dataset = TensorDataset(source_data, source_label,indices)
    source_loader = DataLoader(index_dataset,batch_size=args.batch_size,drop_last= False)
    
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
    indices = torch.arange(len(target_dataset))
    index_dataset = TensorDataset(target_data, target_label,indices)
    target_loader = DataLoader(index_dataset,batch_size=args.batch_size,drop_last= False)
    
    dset_loaders["target"] = target_loader
    return dset_loaders

def cal_acc(loader, fea_bank, socre_bank, netG, netC, args, flag=False):
    start_test = True
    num_sample = len(loader.dataset)
    label_bank = torch.randn(num_sample)  # .cuda()
    pred_bank = torch.randn(num_sample)
    # nu=[]
    # s=[]
    # var_all=[]

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.__next__()
            inputs = data[0]
            labels = data[1]
            indx = data[-1]
            inputs = inputs.cuda()
            fea = (netG(inputs))
            """if args.var:
                var_batch=fea.var()
                var_all.append(var_batch)"""

            # if args.singular:
            # _, ss, _ = torch.svd(fea)
            # s10=ss[:10]/ss[0]
            # s.append(s10)

            outputs = netC(fea)
            softmax_out = nn.Softmax()(outputs)
            # nu.append(torch.mean(torch.svd(softmax_out)[1]))
            output_f_norm = F.normalize(fea)
            # fea_bank[indx] = output_f_norm.detach().clone().cpu()
            label_bank[indx] = labels.float().detach().clone()  # .cpu()
            pred_bank[indx] = outputs.max(-1)[1].float().detach().clone().cpu()
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                # all_fea = output_f_norm.cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                # all_fea = torch.cat((all_fea, output_f_norm.cpu()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )

    _, socre_bank_ = torch.max(socre_bank, 1)
    distance = fea_bank.cpu() @ fea_bank.cpu().T
    _, idx_near = torch.topk(distance, dim=-1, largest=True, k=4)
    score_near = socre_bank_[idx_near[:, :]].float().cpu()  # N x 4

    """acc1 = (score_near.mean(
        dim=-1) == score_near[:, 0]).sum().float() / score_near.shape[0]"""
    acc1 = (
        (score_near.mean(dim=-1) == score_near[:, 0]) & (score_near[:, 0] == pred_bank)
    ).sum().float() / score_near.shape[0]
    acc2 = (
        (score_near.mean(dim=-1) == score_near[:, 0]) & (score_near[:, 0] == label_bank)
    ).sum().float() / score_near.shape[0]

    """if True:
        nu_mean=sum(nu)/len(nu)"""

    # s10_avg=torch.stack(s).mean(0)
    # print('nuclear mean: {:.2f}'.format(nu_mean))

    if True:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = " ".join(aa)
        if True:
            return aacc, acc  # , acc1, acc2#, nu_mean, s10_avg

    else:
        return accuracy * 100, mean_ent


def hyper_decay(x, beta=-2, alpha=1):
    weight = (1 + 10 * x) ** (-beta) * alpha
    return weight



def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netG = Generator(source='CWRU_'+args.s, target='CWRU_'+args.t).cuda()
    netC = Classifier(source='CWRU_'+args.s, target='CWRU_'+args.t).cuda()
    
    modelpath = args.output_dir_src + "/source_G.pt"
    netG.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + "/source_C.pt"
    netC.load_state_dict(torch.load(modelpath))
    param_group = []
    param_group_c = []
    for k, v in netG.named_parameters():
        # if k.find('bn')!=-1:
        if True:
            param_group += [{"params": v, "lr": args.lr * 0.1}]  # 0.1

    for k, v in netC.named_parameters():
        param_group_c += [{"params": v, "lr": args.lr * 1}]  # 1

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    # building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample,64* 300)
    score_bank = torch.randn(num_sample, 3).cuda()

    
    netG.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.__next__()
            inputs = data[0]
            indx = data[-1]
            # labels = data[1]
            inputs = inputs.cuda()
            output = (netG(inputs))
            output_norm = F.normalize(output)
            output_norm = output_norm.view(-1,64*300)
            outputs = netC(output)
            outputs = nn.Softmax(-1)(outputs)
            # print(output_norm.shape)
            # print(fea_bank[indx].shape)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  # .cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    
    netG.train()
    netC.train()
    acc_log = 0

    real_max_iter = max_iter

    while iter_num < real_max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.__next__()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        labels_test = _.cuda()
        if True:
            alpha = (1 + 10 * iter_num / max_iter) ** (-args.beta) * args.alpha
        else:
            alpha = args.alpha

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        features_test = (netG(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        # output_re = softmax_out.unsqueeze(1)

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_norm = output_f_norm.view(-1,64*300)
            output_f_ = output_f_norm.cpu().detach().clone()

            pred_bs = softmax_out

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(
            -1, args.K, -1
        )  # batch x K x C

        loss = torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)
        ) # Equal to dot product
        

        mask = torch.ones((inputs_test.shape[0], inputs_test.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = softmax_out.T  # .detach().clone()#

        dot_neg = softmax_out @ copy  # batch x batch

        dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)
        loss += neg_pred * alpha
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs_test,labels_test)
        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            
            netG.eval()
            netC.eval()
            if args.dset == "visda-2017":
                acc, accc = cal_acc(
                    dset_loaders["target"],
                    fea_bank,
                    score_bank,
                    
                    netG,
                    netC,
                    args,
                    flag=True,
                )
                log_str = (
                    "Task: , Iter:{}/{};  Acc on target: {:.2f}".format(
                         iter_num, max_iter, acc
                    )
                    + "\n"
                    + "T: "
                    + accc
                )

            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")
            
            netG.train()
            netC.train()
            """if acc>acc_log:
                acc_log = acc
                torch.save(
                    netF.state_dict(),
                    osp.join(args.output_dir, "target_F_" + '2021_'+str(args.tag) + ".pt"))
                torch.save(
                    netB.state_dict(),
                    osp.join(args.output_dir,
                                "target_B_" + '2021_' + str(args.tag) + ".pt"))
                torch.save(
                    netC.state_dict(),
                    osp.join(args.output_dir,
                                "target_C_" + '2021_' + str(args.tag) + ".pt"))"""

    return  netG, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPA")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--s", type=str, default='DE', help="source")
    parser.add_argument("--t", type=str, default="FE", help="target")
    parser.add_argument("--max_epoch", type=int, default=15, help="max iterations")
    parser.add_argument("--interval", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument("--dset", type=str, default="visda-2017")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--net", type=str, default="resnet101")
    parser.add_argument("--seed", type=int, default=2021, help="random seed")

    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--output", type=str, default="./model/")
    parser.add_argument("--output_src", type=str, default="./model/")
    parser.add_argument("--tag", type=str, default="LPA")
    parser.add_argument("--da", type=str, default="uda")
    parser.add_argument("--issave", type=bool, default=True)
    parser.add_argument("--cc", default=False, action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--alpha_decay", default=True)
    parser.add_argument("--nuclear", default=False, action="store_true")
    parser.add_argument("--var", default=False, action="store_true")
    parser.add_argument('--class_num',type=int, default=3)
    parser.add_argument('--output_dir_src',default='./model/')
    parser.add_argument("--data_root",type= str, default='./CWRU_dataset/')
    args = parser.parse_args()

    names = ['DE','FE']

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        

        folder = "./data/"
        args.s_dset_path = args.data_root+'CWRU_'+args.s+'.npy'
        args.t_dset_path = args.data_root+'CWRU_'+args.t+'.npy'
        
        # args.output_dir_src = osp.join(
        #     args.output,names[i],
        # )
        args.output_dir = osp.join(
            args.output
        )
        

        if not osp.exists(args.output_dir):
            os.system("mkdir -p " + args.output_dir)
        if not osp.exists(args.output_dir):
             os.mkdir(args.output_dir)

        args.out_file = open(
            osp.join(args.output_dir, "log_{}.txt".format(args.tag)), "w"
        )
        args.out_file.write(print_args(args) + "\n")
        args.out_file.flush()
        train_target(args)
