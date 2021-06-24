import torch
import torchvision

def display_sample_images(loader, writer):
    print("Generating sample input images. View on Tensorboard!")
    """
    loader : PyTorch Dataloader that returns batches of shape (Batch,Time_step,#Channels,Img_h,Img_w)
    writer : tensorboard summary writer
    """

    for idx, (batch,_) in enumerate(loader):
        random_frames = batch[torch.randint(low = 0, high = batch.size(0), size=())]
        img_grid = torchvision.utils.make_grid(random_frames)
        writer.add_image('sample images', img_grid, idx)


def display_model_graph(encoder, decoder, loader, writer):
    print("Generating model graph. View on Tensorboard!")
    class CRNN(torch.nn.Module):
        def __init__(self, encoder, decoder):
            super(CRNN, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            out = self.decoder(self.encoder(x))
            return out
    
    crnn = CRNN(encoder, decoder)
    crnn.cuda()

    dataiter = iter(loader)

    batch,_ = dataiter.next()
    batch = batch.cuda()

    writer.add_graph(crnn, batch)

