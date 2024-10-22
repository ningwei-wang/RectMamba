import argparse

from torchvision.models import resnet50
from tqdm import tqdm
from torch.utils.data import Subset
from torch.optim import SGD
from datasets.dataloader_clothing1m import clothing_dataset
from utils import *
import wandb
import torch.nn as nn
parser = argparse.ArgumentParser('Train with Clothing1M dataset')
parser.add_argument('--dataset_path', metavar='data', default='/data/data/academic/Clothing1M', help='dataset path')

# model settings
parser.add_argument('--theta_s', default=1, type=float, help='Initial threshold for voted correct samples (default: 1)')
parser.add_argument('--theta_r', default=0.99, type=float, help='threshold for relabel samples (default: 0.9)')
parser.add_argument('--lambda_fc', default=1, type=float, metavar='N', help='weight of all data (default: 1)')
parser.add_argument('--k', default=200, type=int, help='neighbors for soft-voting (default: 200)')

# train settings
parser.add_argument('--epochs', default=120, type=int, metavar='N', help='number of total epochs to run (default: 120)')
parser.add_argument('--batch_size', default=20, type=int, help='mini-batch size (default: 32)')
parser.add_argument('--lr', default=0.0003, type=float, help='initial learning rate (default: 0.002)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight_decay', default=1e-3, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', default=3047, type=int, help='seed for initializing training. (default: 3047)')
parser.add_argument('--parallel', default=0, action='store_true', help='Multi-GPU training (default: False)')
parser.add_argument('--gpuid', default='0', type=str, help='Selected GPU (default: "0")')
parser.add_argument('--entity', type=str, help='Wandb user entity')
parser.add_argument('--run_path', type=str, help='run path containing all results')


def adaptive_mixup_coefficient(epoch, max_epochs):
    alpha = 4 * (1 - epoch / max_epochs)
    beta = 4 * (1 - epoch / max_epochs)
    return np.random.beta(alpha, beta)

def train(labeled_trainloader, all_trainloader, encoder, classifier, optimizer, epoch, args):
    encoder.eval()
    classifier.train()
    xlosses = AverageMeter('xloss')
    labeled_train_iter = iter(labeled_trainloader)
    all_bar = tqdm(all_trainloader)
    for [inputs_x1, inputs_x2], labels_x, index in labeled_train_iter:

        batch_size = inputs_x1.size(0)
        inputs_x1, inputs_x2, labels_x = inputs_x1.cuda(), inputs_x2.cuda(), labels_x.cuda()
        targets_x = torch.zeros(batch_size, args.num_classes, device=inputs_x1.device).scatter_(1, labels_x.view(-1, 1),
                                                                                                1)
        l = adaptive_mixup_coefficient(epoch, args.epochs)  # Get mixup coefficient based on current epoch
        l = max(l, 1 - l)

        all_inputs_x = torch.cat([inputs_x1, inputs_x2], dim=0)
        all_targets_x = torch.cat([targets_x, targets_x], dim=0)
        idx = torch.randperm(all_inputs_x.size()[0])
        input_a, input_b = all_inputs_x, all_inputs_x[idx]
        target_a, target_b = all_targets_x, all_targets_x[idx]


        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        logits = classifier(encoder(mixed_input))
        Lce = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        loss = Lce
        xlosses.update(Lce.item())
        all_bar.set_description(
            f'Train epoch {epoch} LR:{optimizer.param_groups[0]["lr"]} Labeled loss: {xlosses.avg:.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(testloader, encoder, classifier, epoch):
    encoder.eval()
    classifier.eval()
    accuracy = AverageMeter('accuracy')
    data_bar = tqdm(testloader)
    with torch.no_grad():
        for i, (data, label, _) in enumerate(data_bar):
            data, label = data.cuda(), label.cuda()
            feat = encoder(data)
            res = classifier(feat)
            pred = torch.argmax(res, dim=1)
            acc = torch.sum(pred == label) / float(data.size(0))
            accuracy.update(acc.item(), data.size(0))
            data_bar.set_description(f'Test epoch {epoch}: Accuracy#{accuracy.avg:.4f}')
    return accuracy.avg


def evaluate(dataloader, model, classifier, args, noisy_label, i):
    model.eval()
    classifier.eval()
    prediction = []
    feature_label = []
    paths = []
    feature_arrbank = []

    with torch.no_grad():
        for (data, target, path) in tqdm(dataloader, desc='Feature extracting'):
            data = data.cuda()
            feature_label.append(target)

            feature = model(data, is_evaluate = True)
            feature_arr = feature.cpu().numpy()
            feature_arrbank.append(feature_arr)

            feature = feature.unsqueeze(-1)
            res = classifier(feature)
            prediction.append(res)
            paths += path

        concatenated_array = np.concatenate(feature_arrbank, axis=0)
        feature_bank = torch.from_numpy(concatenated_array)
        feature_bank = F.normalize(feature_bank, dim=1).to(torch.device("cuda:0"))
        modified_label = torch.cat(feature_label, dim=0)
        modified_label = modified_label.cuda()

        ################################### sample relabelling ###################################
        prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
        his_score, his_label = prediction_cls.max(1)

        def combined_entropy_margin(prediction_cls):
            entropy = -torch.sum(prediction_cls * torch.log(prediction_cls + 1e-5), dim=1)
            sorted_preds, _ = torch.sort(prediction_cls, dim=1, descending=True)
            margin = sorted_preds[:, 0] - sorted_preds[:, 1]
            combined_score = entropy / (margin + 1e-5)
            return combined_score

        combined_score = combined_entropy_margin(prediction_cls)
        top_element = int(combined_score.numel() * (i + 1) / args.epochs)
        threshold = torch.kthvalue(combined_score, top_element).values.item()  # 选取第N小的结合分数作为阈值
        conf_id = (combined_score <= threshold).nonzero(as_tuple=True)[0]
        modified_label = torch.clone(noisy_label).detach()
        modified_label[conf_id] = his_label[conf_id]

        ################################### sample selection ###################################
        prediction_knn = weighted_knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k, 100,
                                      10)  # temperature in weighted KNN
        vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
        vote_max = prediction_knn.max(dim=1)[0]
        right_score = vote_y / vote_max
        clean_id = torch.where(right_score >= args.theta_s)[0]
        noisy_id = torch.where(right_score < args.theta_s)[0]

    return clean_id, noisy_id, modified_label, paths


def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.run_path is None:
        args.run_path = f'Dataset(clothing1m_Model({args.theta_r}_{args.theta_s})'

    global logger
    logger = wandb.init(project='ssr_clothing1m', entity=args.entity, name=args.run_path)
    logger.config.update(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    args.num_classes = 14
    args.image_size = 224

    ################################ Model initialization ###########################################

    # use pretrained encoder
    encoder = resnet50(pretrained=True)
    dim = encoder.fc.in_features

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.001)
            nn.init.constant_(m.bias, 0)

    classifier = torch.nn.Linear(dim, args.num_classes)
    # replace original linear layer
    encoder.fc = torch.nn.Identity()

    classifier.apply(init_weights)

    encoder.cuda()
    classifier.cuda()
    if args.parallel:
        encoder = torch.nn.DataParallel(encoder).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()

    image_dimension = 224
    target_size = (image_dimension, image_dimension)
    ############################# Dataset initialization ##############################################
    # mini-webvision augmentations
    weak_transform = transforms.Compose([
        ResizeAndPad(target_size, 14),
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))
    ])
    strong_transform = transforms.Compose([
        ResizeAndPad(target_size, 14),
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))
    ])
    none_transform = transforms.Compose([
        ResizeAndPad(target_size, 14),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))
    ])

    test_transform = none_transform

    # generate noisy dataset with our transformation
    if not os.path.isdir(f'clothing1m'):
        os.mkdir(f'clothing1m')
    if not os.path.isdir(f'clothing1m/{args.run_path}'):
        os.mkdir(f'clothing1m/{args.run_path}')

    # genarate train dataset with only filtered clean subset
    test_data = clothing_dataset(root_dir=args.dataset_path, transform=test_transform, dataset_mode='test')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size * 10, shuffle=False, num_workers=4,
                                              pin_memory=True)

    #################################### Training initialization #######################################
    optimizer = SGD(
        [{'params': encoder.parameters()}, {'params': classifier.parameters()}],
        lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    acc_logs = open(f'clothing1m/{args.run_path}/acc.txt', 'w')
    stat_logs = open(f'clothing1m/{args.run_path}/stat.txt', 'w')

    save_config(args, f'clothing1m/{args.run_path}')
    print('Train args: \n', args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

    best_acc = 0

    ################################ Training loop ###########################################
    for i in range(args.epochs):
        eval_data = clothing_dataset(root_dir=args.dataset_path, transform=weak_transform, dataset_mode='eval',
                                     num_samples=1000 * args.batch_size)
        eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size * 10, shuffle=False,
                                                  num_workers=4)  # , pin_memory=True)
        clean_id, noisy_id, modified_label, paths = evaluate(eval_loader, encoder, classifier, args, i)

        print(f'Epoch [{i}/{args.epochs}]: clean samples: {len(clean_id)}, noisy samples: {len(noisy_id)}')

        labeled_data = clothing_dataset(root_dir=args.dataset_path, transform=KCropsTransform(strong_transform, 2),
                                        dataset_mode='train', paths=paths, subset=clean_id, labels=modified_label)
        sampler = ClassBalancedSampler(labels=modified_label[clean_id], num_classes=args.num_classes)
        labeled_loader = torch.utils.data.DataLoader(labeled_data, batch_size=args.batch_size, sampler=sampler, drop_last=False,
                                                     num_workers=4)  # , drop_last=True)

        all_data = clothing_dataset(root_dir=args.dataset_path,
                                    transform=MixTransform(strong_transform, weak_transform, 1),
                                    dataset_mode='unlabeled', paths=paths)
        all_loader = torch.utils.data.DataLoader(all_data, batch_size=args.batch_size, num_workers=4, drop_last=False,
                                                 shuffle=True)  # , drop_last=True)

        train(labeled_loader, all_loader, encoder, classifier, optimizer, i, args)

        stat_logs.write(
            f'Epoch [{i}/{args.epochs}]: clean samples: {len(clean_id)}, noisy samples: {len(noisy_id)}\n')
        stat_logs.flush()
        scheduler.step()
        cur_acc = test(test_loader, encoder, classifier, i)
        logger.log({'acc': cur_acc})
        if cur_acc > best_acc:
            best_acc = cur_acc
            save_checkpoint({
                'cur_epoch': i,
                'classifier': classifier.state_dict(),
                'encoder': encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'clothing1m/{args.run_path}/best_acc.pth.tar')
        acc_logs.write(
            f'Epoch [{i}/{args.epochs}]: Best accuracy@1:{best_acc}! Current accuracy@1:{cur_acc}!\n')
        acc_logs.flush()
        print(f'Best accuracy@1:{best_acc}! Current accuracy@1:{cur_acc}!\n')
    save_checkpoint({
        'cur_epoch': i,
        'classifier': classifier.state_dict(),
        'encoder': encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=f'clothing1m/{args.run_path}/last.pth.tar')


if __name__ == '__main__':
    main()
