import torch
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
from PIL import Image
import torchvision.transforms as transforms

class LiquidCrystalDataset(Dataset):
    def __init__(self, data_dir_5cb, data_dir_mbba, transform=None):
        self.data_5cb = glob(os.path.join(data_dir_5cb, '*.jpg'))
        self.data_mbba = glob(os.path.join(data_dir_mbba, '*.jpg'))
        self.data = self.data_5cb + self.data_mbba
        self.labels = [0] * len(self.data_5cb) + [1] * len(self.data_mbba)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--l2', type=float, default=0, help='L2 regularisation')
    parser.add_argument('--aug', action='store_true', default=False, help='Use data augmentation')
    parser.add_argument('--data_path_5cb', type=str, default='./data/5cb', help='Path to 5CB data.')
    parser.add_argument('--data_path_mbba', type=str, default='./data/mbba', help='Path to MBBA data.')
    parser.add_argument('--bond_dim', type=int, default=5, help='MPS Bond dimension')
    parser.add_argument('--nChannel', type=int, default=1, help='Number of input channels')
    parser.add_argument('--dense_net', action='store_true', default=False, help='Using Dense Net model')
    parser.add_argument('--MERA', action='store_true', default=False, help='Using Conv style Tensor Net model')
    parser.add_argument('--kernel', nargs='+', type=int)
    parser.add_argument("--gpu", default=0, help="GPU device ID")
    args = parser.parse_args()

    global_bs = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:'+str(args.gpu))
    print(device)
    with torch.cuda.device('cuda:'+str(args.gpu)):
        batch_size = args.batch_size

        # LoTeNet parameters
        adaptive_mode = False
        periodic_bc   = False

        kernel = args.kernel  # Stride along spatial dimensions
        output_dim = 2 # output dimension

        feature_dim = 2

        logFile = time.strftime("%Y%m%d_%H_%M")+'.txt'
        makeLogFile(logFile)

        normTensor = 0.5*torch.ones(args.nChannel)
        ### Data processing and loading....
        trans_valid = transforms.Compose([transforms.Normalize(mean=normTensor,std=normTensor)])

        if args.aug:
            trans_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(mean=normTensor, std=normTensor)
            ])
            print("Using Augmentation....")
        else:
            trans_train = trans_valid
            print("No augmentation....")

        # Load the 5CB and MBBA datasets
        dataset_train = LiquidCrystalDataset(args.data_path_5cb, args.data_path_mbba, transform=trans_train)
        dataset_valid = LiquidCrystalDataset(args.data_path_5cb, args.data_path_mbba, transform=trans_valid)
        dataset_test = LiquidCrystalDataset(args.data_path_5cb, args.data_path_mbba, transform=trans_valid)

        num_train = len(dataset_train)
        num_valid = len(dataset_valid)
        num_test = len(dataset_test)
        print("Num. train = %d, Num. val = %d" % (num_train, num_valid))

        loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
        loader_valid = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False, drop_last=True)
        loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)

        # Initiliaze input dimensions
        dim = torch.ShortTensor(list(dataset_train[0][0].shape[1:]))
        nCh = int(dataset_train[0][0].shape[0])

        # Initialize the models
        if not args.dense_net:
            if args.MERA:
                print("Using MERA")
                model = MERAnet(input_dim=dim, output_dim=output_dim, nCh=nCh, kernel=kernel, bond_dim=args.bond_dim, feature_dim=feature_dim, adaptive_mode=adaptive_mode, periodic_bc=periodic_bc, virtual_dim=1)
            else:
                print("Using LoTeNet")
                model = loTeNet(input_dim=dim, output_dim=output_dim, nCh=nCh, kernel=kernel, bond_dim=args.bond_dim, feature_dim=feature_dim, adaptive_mode=adaptive_mode, periodic_bc=periodic_bc, virtual_dim=1)
        else:
            print("Densenet Baseline!")
            model = FrDenseNet(depth=40, growthRate=12, reduction=0.5, bottleneck=True, nClasses=output_dim)

        # Choose loss function and optimizer
        loss_fun = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameters:%d" % nParam)
        print(f"Maximum MPS bond dimension = {args.bond_dim}")
        with open(logFile, "a") as f:
            f.write(f"Bond dim: {args.bond_dim}\n")
            f.write(f"Number of parameters: {nParam}\n")

        print(f"Using Adam w/ learning rate = {args.lr:.1e}")
        print(f"Feature_dim: {feature_dim}, nCh: {nCh}, B: {batch_size}")

        model = model.to(device)
        nValid = len(loader_valid)
        nTrain = len(loader_train)
        nTest = len(loader_test)

        maxAuc = 0
        minLoss = 1e3
        convCheck = 5
        convIter = 0

        # Let's start training!
        for epoch in range(args.num_epochs):
            running_loss = 0.
            running_acc = 0.
            t = time.time()
            model.train()

            for i, (inputs, labels) in enumerate(loader_train):
                inputs = inputs.to(device)
                labels = labels.to(device)
                sm = torch.nn.Softmax(dim=1)
                scores = sm(model(inputs))
                preds = scores
                preds = preds.float()
                loss = loss_fun(scores, labels.long())

                with torch.no_grad():
                    running_loss += loss

                # Backpropagate and update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 5 == 0:
                    print(f'Epoch [{epoch + 1}/{args.num_epochs}], Step [{i + 1}/{nTrain}], Loss: {loss.item():.4f}')

            accuracy = computeAuc(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())

            # Evaluate on Validation set
            with torch.no_grad():
                vl_acc, vl_loss = evaluate(loader_valid)
                if vl_acc > maxAuc or vl_loss < minLoss:
                    if vl_loss < minLoss:
                        minLoss = vl_loss
                    if vl_acc > maxAuc:
                        ts_acc, ts_loss = evaluate(loader_test)
                        maxAuc = vl_acc
                        print(f'New Max: {maxAuc:.4f}')
                        print(f'Test Set Loss: {ts_loss:.4f} AUC: {ts_acc:.4f}')
                        with open(logFile, "a") as f:
                            f.write(f'Test Set Loss: {ts_loss:.4f} AUC: {ts_acc:.4f}\n')
                        convEpoch = epoch
                        convIter = 0
                else:
                    convIter += 1
                if convIter == convCheck:
                    if not args.dense_net:
                        print("MPS")
                    else:
