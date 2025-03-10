'''
Visualization of the resulting training data

Terminal:   --logdir= log location --port= port number
tensorboard --logdir=logs          --port=6008

'''

from torch.utils.tensorboard import SummaryWriter
class visual():

    def __init__(self,logs_path):
        self.logs_path = logs_path     
        self.writer = SummaryWriter(self.logs_path)     

    def summary_loss(self,loss,step):
        self.writer.add_scalar("loss", loss, step)  


    def close_summary(self):
        self.writer.close()




