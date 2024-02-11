import torch
import torch.nn as nn

from model.decoder import Decoder
from model.discriminator import Discriminator
from model.encoder import Encoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

batchsize = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def gradient_penalty(discriminator, real, fake):
    t = torch.rand(real.size(0), 1, 1, 1, device=device)
    real = real.to(device)
    fake = fake.to(device)
    interpolates = t * real + (1 - t) * fake
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    disc_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class Hidden:
    def __init__(self):
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.discriminator = Discriminator().to(device)
        # load the model from the file
        if os.path.exists("encoder.pth"):
            self.encoder.load_state_dict(torch.load("encoder.pth"))
        if os.path.exists("decoder.pth"):
            self.decoder.load_state_dict(torch.load("decoder.pth"))
        if os.path.exists("discriminator.pth"):
            self.discriminator.load_state_dict(torch.load("discriminator.pth"))

        self.optimizer_ed = torch.optim.Adam([*self.encoder.parameters(), *self.decoder.parameters()], lr=1e-4)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)
        self.bce_loss = nn.BCELoss().to(device)

    def train(self, epoch, img_path, writter=None, messages=torch.randn(batchsize, 30).to(device)):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        # load the image from the img_path using DataSet
        dataset = TestImageDataset(img_path,
                                   transform=transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]))
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

        for i in range(epoch):
            real = next(iter(dataloader)).to(device)
            # train the discriminator for 10 times
            for k in range(4):
                self.optimizer_d.zero_grad()
                fake = self.encoder(real, messages)
                d_real = self.discriminator(real)
                d_fake = self.discriminator(fake)
                loss_d_real = self.criterion(d_real, torch.ones(batchsize, 1).to(device))
                loss_d_fake = self.criterion(d_fake, torch.zeros(batchsize, 1).to(device))
                gp = gradient_penalty(self.discriminator, real, fake.detach())
                # implement the Gradient Penalty
                loss_d = loss_d_real + loss_d_fake + gp
                loss_d.backward()
                self.optimizer_d.step()
                if k % 5 == 0:
                    print(f"epoch:{i},loss_d:{loss_d.item()},gp:{gp.item()}")
                    if writter is not None:
                        writter.add_scalar("loss_d", loss_d.item(), i)
                        writter.add_scalar("gp", gp.item(), i)
            for k in range(6):
                # train the encoder and decoder
                self.optimizer_ed.zero_grad()
                encoded = self.encoder(real, messages)
                decoded = self.decoder(encoded)
                decoded_probs = torch.sigmoid(decoded)
                d_encoded = self.discriminator(encoded)
                loss_ed = self.criterion(d_encoded, torch.ones(batchsize, 1).to(device))

                # 计算比特精确度损失
                #acc_loss = self.bce_loss(decoded_probs, messages)
                acc_loss = nn.BCEWithLogitsLoss()(decoded, messages)
                # calculate bit accuracy
                bit_acc = 1.0 - torch.mean(torch.abs((decoded_probs > 0.5).float() - messages)).item()
                # bit_acc = 1.0
                # 总损失包括编码器-解码器损失和比特精确度损失
                loss_ed += acc_loss

                loss_ed.backward()
                self.optimizer_ed.step()
                if k % 5 == 0:
                    if writter is not None:
                        writter.add_scalar("loss_ed", loss_ed, i * 2 + k // 5)
                        writter.add_scalar("bit_acc", bit_acc, i * 2 + k // 5)
                    print(f"epoch:{i},loss_ed:{loss_ed.item()},bit_acc:{bit_acc},acc_loss:{acc_loss.item()}")
            if (i+1) % 1000 == 0:
                torch.save(self.encoder.state_dict(), "encoder.pth")
                torch.save(self.decoder.state_dict(), "decoder.pth")
                torch.save(self.discriminator.state_dict(), "discriminator.pth")

