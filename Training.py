import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from snntorch import functional as SF

from snntorch import spikegen

import tqdm



from N_MNIST import *
from Classifier import *

BATCH_SIZE = 64
NUM_THREADS = 7 # set lower! It needs a lot of shared memory

torch.autograd.set_detect_anomaly(True)

def main():

    #
    # Device
    #
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #
    # Augmentation
    #

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    #
    # Dataset
    #
    

    train_ds = N_MNIST(split="Train")
    test_ds = N_MNIST(split="Test")


    #
    # Data loaders
    #

    train_loader = DataLoader(train_ds, 
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_THREADS, 
                              shuffle=True, 
                              drop_last=True,
                              prefetch_factor=16,
                              persistent_workers=False
                        )
    
    test_loader = DataLoader(test_ds, 
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_THREADS, 
                             shuffle=True, 
                             drop_last=True,
                             prefetch_factor=16,
                             persistent_workers=False
                        )

    #
    # Logging
    #

    file_path = f"./logs/"

    writer = SummaryWriter(file_path)

    #
    # Init Model
    #

    model = Classifier()
    model.to(device)

    #
    # Train loop
    #
    for epoch in range(5):
        
        print(f"Epoch {epoch}")

        # Epoch 0 = no training steps are performed 
        # test based on train data
        # -> Determinate initial train_loss and train_accuracy
        if epoch == 0:

            train_loss, train_accuracy = model.test(train_loader, device)

        else:

            model.train()

            for x, targets in tqdm.tqdm(train_loader, position=0, leave=True):

                # x: (bs, t, d) = (64, 20, 1156)
                x = x.permute(1,0,2)

                # x: (t, bs, d) = (20, 64, 1156)

                # Transfer data to GPU (if available)
                x, targets = x.to(device), targets.to(device)

                # Reset gradients
                model.optimizer.zero_grad()

                # Forward pass
                spk_rec = model(x)
                spk_sum = spk_rec.sum(dim=0)
           
                # Calc loss
                loss = model.cce_rate_loss(spk_sum, targets)

                # Backprob
                loss.backward()

                # Update parameters
                model.optimizer.step()

                #
                # Update metrics
                #

                # Loss
                model.loss_metric.update(loss)

                # Accuracy
                accuracy = SF.accuracy_rate(spk_rec, targets)
                model.accuracy_metric.update(accuracy)

       
            train_loss = model.loss_metric.compute()
            train_accuracy = model.accuracy_metric.compute()


        test_loss, test_accuracy = model.test(test_loader, device)

        #
        # Output
        #
        print(f"      train_loss: {train_loss}")
        print(f"       test_loss: {test_loss}")
        print(f"  train_accuracy: {train_accuracy}")
        print(f"   test_accuracy: {test_accuracy}")
 

        #
        # Logging
        #
        writer.add_scalars("Loss",
                            { "Train" : train_loss, "Test" : test_loss },
                            epoch)
        
        writer.add_scalars("Accuracy",
                            { "Train" : train_accuracy, "Test" : test_accuracy },
                            epoch)
        
        
        writer.flush()

        torch.save(model.state_dict(), f"./saved_models/{epoch}")

    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")