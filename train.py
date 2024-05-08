from run_on_patches import split_image_crops
import torch
from utils import save_checkpoint, load_checkpoint,save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import WaterDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from vgg_loss import VGGLoss
from torch.utils.tensorboard import SummaryWriter
import os

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()



def train_fn(
    loader,disc,gen,opt_disc, opt_gen,vgg_loss,l1_loss,bce, g_scaler, d_scaler,writer,tb_step,epoch
):
    loop =  tqdm(loader, leave=True)
 

    for idx, (input_image, target_image) in enumerate(loop):
        x = input_image.to(config.DEVICE)  
        y = target_image.to(config.DEVICE)  

        #Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real) - 0.1 * torch.rand_like(D_real))


            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))

           
            D_loss = D_real_loss + D_fake_loss 
        disc.zero_grad()
        d_scaler.scale(D_loss).backward()  #retain_graph=True
        d_scaler.step(opt_disc)
        d_scaler.update()

        # d_losses.append(D_loss.item())  # Append discriminator loss
        # Train generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = 1e-3 * bce(D_fake, torch.ones_like(D_fake))
            loss_for_vgg =  0.006 * vgg_loss(y_fake, y)
            L1_Loss = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + loss_for_vgg + L1_Loss


        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # g_losses.append(G_loss.item())  # Append generator loss

        writer.add_scalar("Discriminator Loss", D_loss.item(), global_step=tb_step)
        writer.add_scalar("Generator Loss", G_loss.item(), global_step=tb_step)

        
        tb_step += 1


        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
    print(f'Epoch [{epoch+1}], Step [{idx+1}/{len(loader)}], D_loss: {D_loss.item()}, G_loss: {G_loss.item()}')
    return tb_step





def main():
    gen = Generator(in_channels=config.CHANNELS_IMG,features=64).to(config.DEVICE)
    disc = Discriminator(in_channels=config.CHANNELS_IMG).to(config.DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999),)
    opt_gen = optim.Adam(
        gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    writer = SummaryWriter("logs")
    tb_step = 0
    BCE = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    vgg_loss = VGGLoss()
    l1_loss = nn.L1Loss()




    gen.train()
    disc.train()
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    dataset = WaterDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()


    val_dataset = WaterDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        tb_step = train_fn(
            train_loader,disc, gen,opt_disc, opt_gen,vgg_loss,l1_loss,BCE,g_scaler,d_scaler,writer,tb_step,epoch
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen,filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc,filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    try_model = True
    if try_model: 
        gen = Generator(in_channels=3).to(config.DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9)) #(0.0,0.9)

        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        split_image_crops(config.TEST_DIR, gen)
    else: 
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print(torch.cuda.is_available())
        main()
 