import torch


def evaluate(net, dataloader, device,criterion,case):
    net.eval()
    num_val_batches = len(dataloader)
    val_loss = 0

    # iterate over the validation set
    for batch, (Input, Output) in enumerate(dataloader):
        # move images and labels to correct device and type
        Input = Input.to(device=device, dtype=torch.float32)
        Output = Output.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            if 'Aleatoric_He' in case:
                Output_pred,s = net(Input)
                val_loss += criterion(Output_pred, Output,s).data                            
            else:                            
                Output_pred = net(Input)
                val_loss += criterion(Output_pred, Output).data
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return val_loss
    return val_loss / num_val_batches
