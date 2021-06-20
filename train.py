import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from utils.checkpoint import CheckPointer
from utils.display import display_sample_images

from models.encoder import Encoder
from models.decoder import Decoder


from datasets.sbu.train_test_split import train_sets, test_sets # generates K-fold train and test sets
from datasets.sbu.sbu_dataset import  SBU_Dataset

torch.manual_seed(0)

class CFG:
    """
        Defines cofiguration parameters
    """

    # EncoderCNN architecture
    res_size = 224        # ResNet image size
    CNN_embed_dim = 256

    # DecoderRNN architecture
    h_frameLSTM = 256 # hidden state dimension for frame-level LSTM
    h_eventLSTM = 256 # hidden state dimension for event-level LSTM
    num_classes = 8 # number of target category

    # training parameters        
    select_frame = 10   # Select given number of middle frames (left-biased). For sbu-dataset this has to be <=10. 
    num_epochs = 250         # training epochs
    batch_size = 16
    num_workers = 4
    learning_rate = 1e-5
    lr_patience = 10
    log_interval = 1   # interval for displaying training info
    save_model_path = './snapshots'


def evaluate(model, device, loader):
    # set model as testing mode
    encoder, decoder = model
    encoder.eval()
    decoder.eval()

    val_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device).view(-1, )

            output = decoder(encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            val_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    val_loss /= len(loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    val_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    return val_score, val_loss


def train(train_loader, valid_loader):

    encoder = Encoder(CNN_embed_dim = CFG.CNN_embed_dim)
    decoder = Decoder(CNN_embed_dim = CFG.CNN_embed_dim, h_frameLSTM = CFG.h_frameLSTM, h_eventLSTM = CFG.h_eventLSTM, num_classes = CFG.num_classes)

    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

        # Combine all EncoderCNN + DecoderRNN parameters (don't include feature_extractor of encoder (e.g resnet) as we are not training its params)
        crnn_params = list(encoder.module.embedding_layer.parameters()) + list(decoder.parameters())

    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
        encoder.cuda()
        decoder.cuda()
        # Combine all EncoderCNN + DecoderRNN parameters (don't include feature_extractor of encoder (e.g resnet) as we are not training its params)
        crnn_params = list(encoder.embedding_layer.parameters()) + list(decoder.parameters())

    writer = SummaryWriter()
    optimizer = torch.optim.Adam(crnn_params, lr = CFG.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', patience = CFG.lr_patience)
    batch_count = 0


    best_metrics = {
        "accuracy" : -1,
        "val_loss" : 1e6 
    }
    checkpointer = CheckPointer(models = [encoder, decoder], optimizer = optimizer, scheduler = scheduler, 
                                save_dir = "./snapshots", best_metrics = best_metrics, watch_metric = "accuracy")
    # checkpointer.load(use_latest = True)
    
    display_sample_images(train_loader, writer)

    for epoch in range(CFG.num_epochs): 
        encoder.train()
        decoder.train()
        N_count = 0   # counting total trained sample in one epoch
        for batch_idx, (X, y) in enumerate(train_loader):
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            N_count += X.size(0)

            optimizer.zero_grad()
            output = decoder(encoder(X))   # output has dim = (batch, number of classes)

            loss = F.cross_entropy(output, y)

            loss.backward()
            optimizer.step()

            # show information
            if (batch_idx + 1) % CFG.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))

            batch_count += 1
            writer.add_scalar('Loss/train', loss.item(), batch_count)  
        
        train_score, train_loss = evaluate([encoder, decoder], device, train_loader)
        val_score, val_loss = evaluate([encoder, decoder], device, valid_loader)

        print('Train set : Average loss: {:.4f}, Accuracy: {:.6f}\n'.format(train_loss, train_score))
        print('Val set : Average loss: {:.4f}, Accuracy: {:.6f}\n'.format(val_loss, val_score))

        writer.add_scalar(f'Accuracy/train', train_score, epoch + 1)
        writer.add_scalar(f'Accuracy/val', val_score, epoch + 1)
        writer.add_scalar(f'Loss/val', val_loss, epoch + 1)

        scheduler.step(val_score)

        checkpointer.save_checkpoint(current_metrics = {"accuracy" : val_score, "val_loss" : val_loss})
    
    print(f'best metrics {checkpointer.best_metrics}')


def main():
    transform = transforms.Compose([transforms.Resize([CFG.res_size, CFG.res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    use_cuda = torch.cuda.is_available()                   # check if GPU exists

    # using first fold
    train_set = SBU_Dataset(train_sets[0], CFG.select_frame, mode='train', transform=transform)
    valid_set = SBU_Dataset(test_sets[0], CFG.select_frame, mode='valid', transform=transform)

    params = {'batch_size': CFG.batch_size, 'shuffle': True, 'num_workers': CFG.num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    train(train_loader, valid_loader)


if __name__ == "__main__":
    main()

    
