import os
import timeit
from datetime import datetime
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
from utils.display import display_sample_images, display_model_graph
from utils.training_utils import repeat_k_times

from models.models import Model1

from datasets.sbu.train_test_split import train_sets, test_sets # generates K-fold train and test sets
from datasets.sbu.sbu_dataset import  SBU_Dataset

# torch.manual_seed(0)

# import argparse
# parser = argparse.ArgumentParser(description='Training for multiple runs')
# parser.add_argument('--run',type=int, help='run number', required = True)
# parser.add_argument('--fold',type=int, help='fold number', required = True)
# args = parser.parse_args()

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
    num_epochs = 200         # training epochs
    batch_size = 16
    num_workers = 4
    learning_rate = 1e-5
    lr_patience = 10
    log_interval = 1   # interval for displaying training info

    save_checkpoint = True
    save_tensorboard = True
    save_model_path = '/usr/local/data01/rohitram/hpc-snapshots/sbu_snapshots_5times_xavier_normal'
    save_tensorboard_dir = '/usr/local/data01/rohitram/hpc-snapshots/runs_5times_xavier_normal'


def evaluate(model, device, loader):
    # set model as testing mode
    model.eval()

    val_loss = 0
    # all_y = []
    # all_y_pred = []

    right, total = 0,0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device).view(-1, )

            output = model(X)

            loss = F.cross_entropy(output, y, reduction='sum')
            val_loss += loss.item()                 # sum up batch loss
            y_pred = torch.argmax(output, axis=1)  #  get the index of the max log-probability

            # collect all y and y_pred in all batches
            right += torch.sum(y_pred == y)
            total += y_pred.shape[0]

    val_loss /= len(loader.dataset)

    # compute accuracy
    # all_y = torch.stack(all_y, dim=0)
    # all_y_pred = torch.stack(all_y_pred, dim=0)
    val_score = right * 1.0 / total

    return val_score, val_loss


@repeat_k_times(5)
def train(train_set, valid_set, fold_no = None, run_no = None):

    # starting_seed = torch.seed()

    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    save_model_subdir = ""
    if fold_no is not None:
        save_model_subdir =  os.path.join(save_model_subdir, f"fold={fold_no}")
    if run_no is not None:
        save_model_subdir =  os.path.join(save_model_subdir, f"run={run_no}")


    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    default_log_dir = current_time

    save_model_dir = os.path.join(CFG.save_model_path, save_model_subdir)
    save_tensorboard_dir = os.path.join(CFG.save_tensorboard_dir, save_model_subdir, default_log_dir)
    
    params = {'batch_size': CFG.batch_size, 'shuffle': True, 'num_workers': CFG.num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    model = Model1(CNN_embed_dim =  CFG.CNN_embed_dim, h_frameLSTM = CFG.h_frameLSTM, h_eventLSTM = CFG.h_eventLSTM, num_classes = CFG.num_classes).to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

        # Combine all EncoderCNN + DecoderRNN parameters (don't include feature_extractor of encoder (e.g resnet) as we are not training its params)
        crnn_params = list(model.module.encoder.embedding_layer.parameters()) + list(model.module.frameLSTM.parameters()) \
                    + list(model.module.eventLSTM.parameters()) + list(model.module.fc.parameters())
    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
        # Combine all EncoderCNN + DecoderRNN parameters (don't include feature_extractor of encoder (e.g resnet) as we are not training its params)
        crnn_params = list(model.encoder.embedding_layer.parameters()) + list(model.frameLSTM.parameters()) \
                    + list(model.eventLSTM.parameters()) + list(model.fc.parameters())

    if CFG.save_tensorboard:
        writer = SummaryWriter(log_dir = save_tensorboard_dir)

    optimizer = torch.optim.Adam(crnn_params, lr = CFG.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', patience = CFG.lr_patience)
    batch_count = 0


    best_metrics = {
        "val_accuracy" : -1,
        "val_loss" : 1e6 
    }
    if not CFG.save_checkpoint:
        save_model_dir = None
    checkpointer = CheckPointer(models = [model], optimizer = optimizer, scheduler = scheduler, 
                                save_dir = save_model_dir, best_metrics = best_metrics, watch_metric = "val_accuracy")
    # checkpointer.load_checkpoint(checkpoint_type = "last")
    
    # display_sample_images(train_loader, writer)
    # display_model_graph(encoder, decoder, train_loader, writer)

    # train_log_file = open(os.path.join(save_model_dir, "train_logs.txt"), "w")
    # train_log_file.write(f'starting pytorch seed is {starting_seed}\n\n')


    for epoch in range(CFG.num_epochs): 
        model.train()
        N_count = 0   # counting total trained sample in one epoch
        epoch_start_time = timeit.default_timer()

        for batch_idx, (X, y) in enumerate(train_loader):
            batch_start_time = timeit.default_timer()
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            N_count += X.size(0)

            optimizer.zero_grad()
            output = model(X)   # output has dim = (batch, number of classes)

            loss = F.cross_entropy(output, y)

            loss.backward()
            optimizer.step()

            batch_count += 1

            if CFG.save_tensorboard:
                writer.add_scalar('Loss/train', loss.item(), batch_count)  

            batch_end_time = timeit.default_timer()

            # show information
            if (batch_idx + 1) % CFG.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime:{:.3f}'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), batch_end_time - batch_start_time))     
            # train_log_file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
            #         epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))



         
        
        train_score, train_loss = evaluate(model, device, train_loader)
        val_score, val_loss = evaluate(model, device, valid_loader)

        print('\nTrain set : Average loss: {:.4f}, Accuracy: {:.6f}'.format(train_loss, train_score))
        print('Val set : Average loss: {:.4f}, Accuracy: {:.6f}\n'.format(val_loss, val_score))
        # train_log_file.write('\nTrain set : Average loss: {:.4f}, Accuracy: {:.6f}\n'.format(train_loss, train_score))
        # train_log_file.write('Val set : Average loss: {:.4f}, Accuracy: {:.6f}\n'.format(val_loss, val_score))

        if CFG.save_tensorboard:
            writer.add_scalar(f'Accuracy/train', train_score, epoch + 1)
            writer.add_scalar(f'Accuracy/val', val_score, epoch + 1)
            writer.add_scalar(f'Loss/val', val_loss, epoch + 1)

        scheduler.step(val_score)

        checkpointer.save_checkpoint(current_metrics = {"val_accuracy" : val_score, "val_loss" : val_loss})
    
        epoch_end_time = timeit.default_timer()  

        print(f'Epoch run time : {epoch_end_time-epoch_start_time:.3f}\n')
    # train_log_file.close()
    print(f'best metrics {checkpointer.best_metrics}')

    return checkpointer.best_metrics


def main():
    transform = transforms.Compose([transforms.Resize([CFG.res_size, CFG.res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    for fold_no in range(5):
        train_set = SBU_Dataset(train_sets[fold_no], CFG.select_frame, mode='train', transform=transform, fold_no = fold_no+1)
        valid_set = SBU_Dataset(test_sets[fold_no], CFG.select_frame, mode='valid', transform=transform, fold_no = fold_no+1)

        result_metrics = train(train_set = train_set, valid_set = valid_set, fold_no = fold_no + 1)
        with open(os.path.join(CFG.save_model_path, f"fold={fold_no+1}", "best_results.txt"), "w") as f:
            for key,val in result_metrics.items():
                f.write(f"{key} : {val}\n")

    # fold_no = int(args.fold) 
    # run_no = int(args.run)

    # train_set = SBU_Dataset(train_sets[fold_no], CFG.select_frame, mode='train', transform=transform, fold_no = fold_no+1)
    # valid_set = SBU_Dataset(test_sets[fold_no], CFG.select_frame, mode='valid', transform=transform, fold_no = fold_no+1)

    # result_metrics = train(train_set = train_set, valid_set = valid_set, fold_no = fold_no + 1, run_no = args.run)
    # with open(os.path.join(CFG.save_model_path, f"fold={fold_no+1}", "best_results.txt"), "w") as f:
    #     for key,val in result_metrics.items():
    #         f.write(f"{key} : {val}\n")


if __name__ == "__main__":
    main()

    
