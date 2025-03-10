import dist_util
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import Mydata
import matplotlib.pyplot as plt
import numpy as np
import unet_model_fancon_dilaion
# import visual_data
import pandas as pd



def main():
    dist_util.setup_dist()

    # Model target datapath
    traget_path = ""
    # The data used to train the model
    train_path = ""
    # The save path for the model
    model_save_path = ""
    # CSV file to save the model loss function
    model_loss_save_path = ""
    # tensorboard file save path
    tensorboard_save_path = ""


    model = unet_model_fancon_dilaion.UNet(n_channels=1, n_classes=1)
    model.to(dist_util.dev())
    # Create data set
    traget_data = Mydata(
        trg_dir = traget_path,
        tra_dir= train_path,
        batch_size= 12,
        image_size = 64,
        category = 'circle'         #circle、rec、lung .If it is' lung ', it is constructed according to the data format of the simulated lung
    )
    

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    # vis = visual_data.visual(tensorboard_save_path)
    num_epochs = 200
    loss_end = np.zeros(num_epochs+1)
    for epoch in range(num_epochs):
        measure,distribution= next(traget_data)
        outputs = model(measure.to(dist_util.dev()))
        outputs.to(dist_util.dev())
        loss = criterion(outputs, distribution.to(dist_util.dev()))
        loss.to(dist_util.dev())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        temp_loss = loss.cpu()
        temp = float(temp_loss)
        loss_end[epoch] = temp
        # vis.summary_loss(loss,epoch)
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # After the training is complete, save the model
    torch.save(model.state_dict(), model_save_path) 
    # vis.close_summary()
    loss_csv = pd.DataFrame(loss_end)
    loss_csv.to_csv(model_loss_save_path)



if __name__ == "__main__":
    main()


