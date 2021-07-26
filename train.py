import os
import numpy as np
import timeit
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
import hydra

from utils.checkpoint import CheckPointer, get_save_directory
from utils.display import display_sample_images, display_model_graph
from utils.training_utils import repeat_k_times
from models.models import Model1, Model2, Model3, Model4
from datasets.sbu.train_test_split import train_sets, test_sets # generates K-fold train and test sets
from datasets.sbu.sbu_dataset import  M1_SBU_Dataset, M2_SBU_Dataset, M3_SBU_Dataset, M4_SBU_Dataset

def evaluate(model, device, loader):
    model.eval()
    val_loss = 0
    right, total = 0,0
    with torch.no_grad():
        for *inps, y in loader:
            for i in range(len(inps)):
                inps[i] = inps[i].to(device)
            y = y.to(device).view(-1, )

            output = model(*inps)

            loss = F.cross_entropy(output, y, reduction='sum')
            val_loss += loss.item()                 
            y_pred = torch.argmax(output, axis=1)  

            right += torch.sum(y_pred == y)

    val_loss /= len(loader.dataset)
    val_score = right * 1.0 / len(loader.dataset)

    return val_score, val_loss


def train(CFG, train_set, valid_set, save_model_subdir, fold_no = None, run_no = None):

    if CFG.training.deterministic:
        # enforce full determinism
        torch.manual_seed(CFG.training.pytorch_seed)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(True)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    use_cuda = torch.cuda.is_available()                  
    device = torch.device("cuda" if use_cuda else "cpu")  

    if fold_no is not None:
        save_model_subdir = os.path.join(save_model_subdir, f"fold={fold_no}")
    if run_no is not None:
        save_model_subdir = os.path.join(save_model_subdir, f"run={run_no}")

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    save_model_dir = os.path.join(
        CFG.training.save_model_path, save_model_subdir, current_time
    )
    save_tensorboard_dir = os.path.join(
        CFG.training.save_tensorboard_dir, save_model_subdir, current_time
    )
    
    params = CFG.training.dataloader if use_cuda else {}
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    if CFG.training.model == 'model1':
        model = Model1(**CFG.model1).to(device)
    elif CFG.training.model == 'model2':
        model = Model2(**CFG.model2).to(device)
    elif CFG.training.model == 'model3':
        model = Model3(**CFG.model3).to(device)
    elif CFG.training.model == 'model4':
        model = Model4(**CFG.model4).to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

        if CFG.training.model == 'model1':
        # Combine all EncoderCNN + DecoderRNN parameters (don't include feature_extractor of encoder (e.g resnet) as we are not training its params)
            crnn_params = list(model.module.encoder.embedding_layer.parameters()) + list(model.module.frameLSTM.parameters()) \
                        + list(model.module.eventLSTM.parameters()) + list(model.module.fc.parameters())
        elif CFG.training.model == 'model2':
            crnn_params = model.module.parameters()
        elif CFG.training.model == 'model3':
            crnn_params = model.module.parameters()
        elif CFG.training.model == 'model4':
            crnn_params = list(model.module.encoder.embedding_layer.parameters()) + list(model.module.frameLSTM.parameters()) \
                        + list(model.module.eventLSTM.parameters()) + list(model.module.attention.parameters()) \
                        + list(model.module.fc.parameters())

    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
        # Combine all EncoderCNN + DecoderRNN parameters (don't include feature_extractor of encoder (e.g resnet) as we are not training its params)
        if CFG.training.model == 'model1':
            crnn_params = list(model.encoder.embedding_layer.parameters()) + list(model.frameLSTM.parameters()) \
                        + list(model.eventLSTM.parameters()) + list(model.fc.parameters())
        elif CFG.training.model == 'model2':
            crnn_params = model.parameters()
        elif CFG.training.model == 'model3':
            crnn_params = model.parameters()
        elif CFG.training.model == 'model4':
            crnn_params = list(model.encoder.embedding_layer.parameters()) + list(model.frameLSTM.parameters()) \
                        + list(model.eventLSTM.parameters()) + list(model.attention.parameters()) \
                        + list(model.fc.parameters())

    if CFG.training.save_tensorboard:
        writer = SummaryWriter(log_dir = save_tensorboard_dir)

    optimizer = torch.optim.Adam(crnn_params, lr = CFG.training.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', patience = CFG.training.lr_patience)
    batch_count = 0


    best_metrics = {
        "val_accuracy" : -1,
        "val_loss" : 1e6 
    }
    if not CFG.training.save_checkpoint:
        save_model_dir = None
    checkpointer = CheckPointer(models = [model], optimizer = optimizer, scheduler = None, 
                                save_dir = save_model_dir, best_metrics = best_metrics, watch_metric = "val_accuracy")
    
    # display_sample_images(train_loader, writer)
    # display_model_graph(encoder, decoder, train_loader, writer)

    for epoch in range(CFG.training.num_epochs): 
        model.train()
        N_count = 0   
        epoch_start_time = timeit.default_timer()
        
        for batch_idx, (*inps,y) in enumerate(train_loader):
            batch_start_time = timeit.default_timer()

            for i in range(len(inps)):
                inps[i] = inps[i].to(device)
            y = y.to(device).view(-1, )

            N_count += inps[0].size(0)
            optimizer.zero_grad()

            output = model(*inps)  
            loss = F.cross_entropy(output, y)

            loss.backward()
            optimizer.step()

            batch_count += 1

            if CFG.training.save_tensorboard:
                writer.add_scalar('Loss/train', loss.item(), batch_count)  

            batch_end_time = timeit.default_timer()

            if (batch_idx + 1) % CFG.training.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime:{:.3f}'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), batch_end_time - batch_start_time))      
        
        train_score, train_loss = evaluate(model, device, train_loader)
        val_score, val_loss = evaluate(model, device, valid_loader)
        print('\nTrain set : Average loss: {:.4f}, Accuracy: {:.6f}'.format(train_loss, train_score))
        print('Val set : Average loss: {:.4f}, Accuracy: {:.6f}\n'.format(val_loss, val_score))
        if CFG.training.save_tensorboard:
            writer.add_scalar(f'Accuracy/train', train_score, epoch + 1)
            writer.add_scalar(f'Accuracy/val', val_score, epoch + 1)
            writer.add_scalar(f'Loss/val', val_loss, epoch + 1)
        # scheduler.step(val_score)
        checkpointer.save_checkpoint(current_metrics = {"val_accuracy" : val_score, "val_loss" : val_loss})
        epoch_end_time = timeit.default_timer()  
        print(f'Epoch run time : {epoch_end_time-epoch_start_time:.3f}\n')

    print(f'best metrics {checkpointer.best_metrics}')

    return checkpointer.best_metrics

@hydra.main(config_path="configs", config_name="config")
def main(DEF_CFG):
    CFG = DEF_CFG.dataset

    global train
    if not CFG.training.deterministic:
        num_runs = CFG.training.num_runs
        decorator = repeat_k_times(num_runs)
        train = decorator(train)

    for fold_no in CFG.training.folds:
        if CFG.training.model == 'model1':
            train_set = M1_SBU_Dataset(train_sets[fold_no], CFG.training.select_frame, mode='train', resize=CFG.model1.resize, fold_no = fold_no+1)
            valid_set = M1_SBU_Dataset(test_sets[fold_no], CFG.training.select_frame, mode='valid', resize=CFG.model1.resize, fold_no = fold_no+1)
        elif CFG.training.model == 'model2':
            train_set = M2_SBU_Dataset(CFG.model2.pose_coord, train_sets[fold_no], CFG.training.select_frame, mode='train', resize=CFG.model1.resize, fold_no = fold_no+1)
            valid_set = M2_SBU_Dataset(CFG.model2.pose_coord, test_sets[fold_no], CFG.training.select_frame, mode='valid', resize=CFG.model1.resize, fold_no = fold_no+1)
        elif CFG.training.model == 'model3':
            train_set = M3_SBU_Dataset(CFG.model3.pose_coord, train_sets[fold_no], CFG.training.select_frame, mode='train', resize=CFG.model1.resize, fold_no = fold_no+1)
            valid_set = M3_SBU_Dataset(CFG.model3.pose_coord, test_sets[fold_no], CFG.training.select_frame, mode='valid', resize=CFG.model1.resize, fold_no = fold_no+1)
        elif CFG.training.model == 'model4':
            train_set = M4_SBU_Dataset(CFG.model3.pose_coord, train_sets[fold_no], CFG.training.select_frame, mode='train', resize=CFG.model1.resize, fold_no = fold_no+1)
            valid_set = M4_SBU_Dataset(CFG.model3.pose_coord, test_sets[fold_no], CFG.training.select_frame, mode='valid', resize=CFG.model1.resize, fold_no = fold_no+1)
        else:
            raise Exception(f"invalid model name - {CFG.training.model}! Must be one of model1, model2, model3, model4")

        save_model_subdir = get_save_directory(CFG)

        result_metrics = train(CFG = CFG, train_set = train_set, valid_set = valid_set, save_model_subdir = save_model_subdir, fold_no = fold_no + 1)
        
        with open(os.path.join(CFG.training.save_model_path, save_model_subdir, f"fold={fold_no+1}", "best_results.txt"), "w") as f:
            for key,val in result_metrics.items():
                f.write(f"{key} : {val}\n")


if __name__ == "__main__":
    main()

    
