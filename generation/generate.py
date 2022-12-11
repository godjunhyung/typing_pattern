import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from data import Password, Password_1, Password_10, Password_20, Password_30
import argparse
from model import Generator, Discriminator
from trainer import run_epoch, val_epoch

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False)
    args = parser.parse_args()
    return args

def main(args):
    torch.manual_seed(0)
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    num_classes = [i for i in range(100)]
    pass_len = 20
    batch_size = 128
    learning_rate = 0.001
    beta1, beta2 = 0.5, 0.999
    num_epochs = 100

    if args.train == "True":
        composed = transforms.Compose([transforms.ToTensor()])
        tr_dataset = Password(num_classes, True, pass_len, composed)
        tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size,shuffle=True,drop_last=True, num_workers=8)
        te_dataset = Password_30(num_classes, False, pass_len, composed)
        te_dataloader = DataLoader(te_dataset, batch_size=batch_size,shuffle=False,drop_last=False, num_workers=8)

        model_gen = Generator(pass_len, batch_size).to(device)
        model_dis = Discriminator(pass_len).to(device)
        opt_gen = optim.Adam(model_gen.parameters(), lr=learning_rate, betas=(beta1,beta2))
        opt_dis = optim.Adam(model_dis.parameters(), lr=learning_rate, betas=(beta1,beta2))

        for epoch in range(num_epochs):
            g_loss, d_loss = run_epoch(model_gen, model_dis, opt_gen, opt_dis, tr_dataloader, device)
            val_acc = val_epoch(model_gen, te_dataloader, True)

            print(f"Epoch: {epoch} \
            g_loss: {g_loss}, d_loss: {d_loss}, val_acc: {val_acc}")

    else:
        composed = transforms.Compose([transforms.ToTensor()])
        te_dataset_1 = Password_1(num_classes, False, pass_len, composed)
        te_dataloader_1 = DataLoader(te_dataset_1, batch_size=batch_size,shuffle=False,drop_last=False,num_workers=4)
        te_dataset_10 = Password_10(num_classes, False, pass_len, composed)
        te_dataloader_10 = DataLoader(te_dataset_10, batch_size=batch_size,shuffle=False,drop_last=False,num_workers=4)
        te_dataset_20 = Password_20(num_classes, False, pass_len, composed)
        te_dataloader_20 = DataLoader(te_dataset_20, batch_size=batch_size,shuffle=False,drop_last=False,num_workers=4)
        te_dataset_30 = Password_30(num_classes, False, pass_len, composed)
        te_dataloader_30 = DataLoader(te_dataset_30, batch_size=batch_size,shuffle=False,drop_last=False,num_workers=4)

        model_gen = Generator(pass_len, batch_size).to(device)
        chk = torch.load("ckpt/model.pth")
        model_gen.load_state_dict(chk)

        val_acc = val_epoch(model_gen, te_dataloader_1)
        print(f"TOP-1: {val_acc}")
        val_acc = val_epoch(model_gen, te_dataloader_10)
        print(f"TOP-10: {val_acc}")
        val_acc = val_epoch(model_gen, te_dataloader_20)
        print(f"TOP-20: {val_acc}")
        val_acc = val_epoch(model_gen, te_dataloader_30)
        print(f"TOP-30: {val_acc}")

if __name__=="__main__":
    args = parse()
    main(args)