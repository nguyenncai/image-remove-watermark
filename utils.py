from subprocess import check_output
import torch
import config
from torchvision.utils import save_image
from run_on_patches import split_image_crops
import matplotlib.pyplot as plt



def save_checkpoint(model, optimizer,filename="checkpoint.pth"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint,filename)


# def plot_losses(d_losses, g_losses, epoch, folder='loss'):
#     plt.figure(figsize=(10,5))
#     plt.title("Discriminator and Generator Loss During Training")
#     plt.plot(d_losses, label='D Loss')
#     plt.plot(g_losses, label='G Loss')
#     plt.xlabel('iterations')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(f'{{folder}}/loss_epoch_{{epoch}}.png')
#     plt.close()



def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_some_examples(generator: object,
                  val_loader: object,
                  epoch: int,
                  folder: str
) -> None:
   
    input_image, target_image = next(iter(val_loader))
    input_image = input_image.to(config.DEVICE)
    target_image = target_image.to(config.DEVICE)
    generator.eval()
    
    with torch.no_grad():
        generated_image = generator(input_image)
        generated_image = generated_image * 0.5 + 0.5     # Removes normalization.
        save_image(generated_image, folder + f"/generated_image_{epoch}.jpg")
        # save_image(input_image * 0.5 + 0.5, folder + f"/input_image_{epoch}.png")
        # save_image(target_image * 0.5 + 0.5, folder + f"/target_image_{epoch}.png")
    
        
        # if epoch == 1:
        #     save_image(target_image * 0.5 + 0.5, folder + f"/target_image_{epoch}.png")
    
    generator.train()


