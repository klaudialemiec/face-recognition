from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from ImageDataset import ImageDataset
from SiameseNetwork import SiameseNetwork
from ContrastiveLoss import ContrastiveLoss
from utils import show_plot

from siamese_model import siamese_network

class Config:
    training_dir = "/home/hoang/comvis/datasets/at_t/orl_faces"
    testing_dir = "/home/hoang/comvis/datasets/at_t/test"
    train_batch_size = 32
    train_num_epochs = 50

transforms = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

folder_dataset = datasets.ImageFolder(root=Config.training_dir)
siamese_dataset = ImageDataset(imageFolderDataset=folder_dataset,
                               transform=transforms, should_invert=False)

train_dataloader = DataLoader(dataset=siamese_dataset, num_workers=2,
                              shuffle=True, batch_size=Config.train_batch_size)

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.005)

counter = []
loss_history = []
iteration_number = 0

triplets_generator = siamese_network.prepare_triplets()

def main():
    global iteration_number
    for epoch in range(Config.train_num_epochs):

        for triplet in triplets_generator:
            anchor, positive, negative = next(triplets_generator)
            anchor, positive, negative = Variable(anchor).cuda(), Variable(positive).cuda(), Variable(negative).cuda()
            optimizer.zero_grad()
            anchor_emb = net(anchor)
            positive_emb = net(positive)
            negative_emb = net(negative)

            contrastive_loss = criterion(output1, output2, label)


        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            contrastive_loss = criterion(output1, output2, label)
            contrastive_loss.backward()
            optimizer.step()
            print ("training epoch {} / current loss {}".format(epoch, contrastive_loss.data.cpu().numpy().item()))
            iteration_number += 1
            counter.append(iteration_number)
            loss_history.append(contrastive_loss.data.cpu().numpy().item())

    show_plot(counter, loss_history)


if __name__=="__main__":
    main()