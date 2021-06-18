import torch
import torchvision

def display_sample_images(loader, writer):
    print("Generating sample input images. View on Tensorboard!")
    """
    loader : PyTorch Dataloader that returns batches of shape (Batch,Time_step,#Channels,Img_h,Img_w)
    writer : tensorboard summary writer
    """
    batch_size = loader.batch_size

    for idx, (batch,_) in enumerate(loader):
        random_frames = batch[torch.randint(low = 0, high = batch_size, size=())]
        img_grid = torchvision.utils.make_grid(random_frames)
        writer.add_image('sample images', img_grid, idx)