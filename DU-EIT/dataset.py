import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math
import dist_util



class Create_Eit_DataModel():

    def __init__(self,coord, background, data_row_col, radius):
        self.coord = coord
        self.background = background
        self.data_row_col = data_row_col
        self.radius = int(radius * (self.data_row_col // 12))

    def distance(self,coord1,coord2):

        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)



    def Create_Eit_DataModel(self):
        '''
            The corresponding conductivity distribution is created using coordinates
            coord: Anomalous area coordinates
            background: Background conductivity

        '''
        abnormal_num= self.coord.shape[0]                                   # Obtain the number of abnormal areas
        classes_flag = self.coord[0].shape[0]  
        DataModel = np.zeros((self.data_row_col, self.data_row_col))        # Create the initial conductivity distribution matrix
        certer = [self.data_row_col//2,self.data_row_col//2]                # Calculate the center point coordinates 
        # The circular shell area is set to 1
        for i in range(0,self.data_row_col):
            for j in range(0,self.data_row_col):
                dis = self.distance([i,j],certer)
                if dis <= self.data_row_col / 2:
                    DataModel[i][j] = 1     
        

        if classes_flag == 4:
            for i in range(0,abnormal_num):  
                self.coord[i][1] = 0 -  self.coord[i][1] 
                temp0 = ((6.0 + self.coord[i][0]) / 12.0) * self.data_row_col
                temp1 = ((6.0 + self.coord[i][1]) / 12.0) * self.data_row_col
                self.coord[i][1] = temp0
                self.coord[i][0] = temp1
                radius = int(self.coord[i][3] * (self.data_row_col // 12))
                self.coord[i][3] = radius

            for i in range(0,abnormal_num):
                for j in range(0,self.data_row_col):
                    for k in range(0,self.data_row_col):
                        dis = self.distance([j,k],self.coord[i])
                        if dis < self.coord[i][3]:
                            DataModel[j][k] = self.coord[i][2]
            DataModel = torch.Tensor(DataModel)
            DataModel = torch.unsqueeze(DataModel, 0)
            DataModel = DataModel.numpy()
            DataModel = DataModel.reshape(-1)
        elif classes_flag == 5:       
            for i in range(0,abnormal_num):
                # It is currently a circular anomaly area
                if self.coord[i][4] == 0:
                    self.coord[i][1] = 0 -  self.coord[i][1] 
                    temp0 = ((6.0 + self.coord[i][0]) / 12.0) * 64.0
                    temp1 = ((6.0 + self.coord[i][1]) / 12.0) * 64.0
                    self.coord[i][1] = temp0
                    self.coord[i][0] = temp1
                    radius = int(self.coord[i][3] * (self.data_row_col // 12))
                    self.coord[i][3] = radius
                    for j in range(0,self.data_row_col):
                        for k in range(0,self.data_row_col):
                            dis = self.distance([j,k],self.coord[i])
                            if dis < self.coord[i][3]:
                                DataModel[j][k] = self.coord[i][2]

                if self.coord[i][4] != 0 or self.coord[i][3] != 0:
                    self.coord[i][1] = 0 -  self.coord[i][1] 
                    temp0 = ((6.0 + self.coord[i][0]) / 12.0) * 64.0
                    temp1 = ((6.0 + self.coord[i][1]) / 12.0) * 64.0
                    self.coord[i][1] = temp1
                    self.coord[i][0] = temp0

                    self.coord[i][3] = 0 -  self.coord[i][3] 
                    temp2 = ((6.0 + self.coord[i][2]) / 12.0) * 64.0
                    temp3 = ((6.0 + self.coord[i][3]) / 12.0) * 64.0
                    self.coord[i][3] = temp3
                    self.coord[i][2] = temp2
                    for j in range(0,self.data_row_col):
                        for k in range(0,self.data_row_col):
                            if j > self.coord[i][0] and j < self.coord[i][2]:
                                if k < self.coord[i][1] and k > self.coord[i][3]:
                                    DataModel[k][j] = self.coord[i][4]
            DataModel = torch.Tensor(DataModel)
            DataModel = torch.unsqueeze(DataModel, 0)
            DataModel = DataModel.numpy()
            DataModel = DataModel.reshape(-1)            
        return DataModel
 

def Fusion_splicing(image,image_size):

    # row = image_size // 8
    # col = image_size // 8  
    temp01 = (image[:,0] + image[:,1])/2
    temp12 = (image[:,1] + image[:,2])/2
    temp23 = (image[:,2] + image[:,3])/2
    temp34 = (image[:,3] + image[:,4])/2
    cat2_data0 = (temp01 + temp12)/2
    cat2_data1 = (temp12 + temp23)/2
    cat2_data2 = (temp23 + temp34)/2
    cat2_data = np.zeros((8,3))
    cat2_data[:,0] = cat2_data0
    cat2_data[:,1] = cat2_data1
    cat2_data[:,2] = cat2_data2
    end_data = np.zeros((8,8))
    end_data[:,:5] = image
    end_data[:,5:8] = cat2_data
    return end_data
 

class MyDateset(Dataset):
    ''' 
        Prepare your own data set, and the data input is CSV format
        measure_path: Is the path to the data
    '''
    def __init__(self, trg_path,tra_path,image_size,category = 'circle'):
        self.total = 0
        self.category = category
        self.sample_list = list()
        self.trg_path = trg_path
        self.tra_path = tra_path
        self.image_size = image_size
        for json_data in os.listdir(self.trg_path):  
             self.total += 1

        for json_data in os.listdir(self.trg_path):  
            self.sample_list.append(json_data)
        
    
    def __len__(self):
        return self.total

    def __getitem__(self, index):

        item = self.sample_list[index]
        
        if self.category == 'lung':

            # Create conductivity distribution data (lung target data)
            distribution_data = pd.read_csv(self.trg_path + '//' + item)
            distribution_data = distribution_data.to_numpy()
            temp = np.zeros((64,64))
            temp[:1,26:38]=1
            temp[1:64,:] = distribution_data
            data = torch.tensor(temp)
            data = torch.unsqueeze(data,0)
            data = data.numpy()
        else:
            # Create conductivity distribution data (circular, matrix target data)
            distribution_data = pd.read_csv(self.trg_path + '//' + item)
            distribution_data = distribution_data.to_numpy()
            distribution_data = distribution_data.T
            datamodel = Create_Eit_DataModel(distribution_data,1,self.image_size,1)
            data = datamodel.Create_Eit_DataModel()
            data = np.reshape(data,(self.image_size,self.image_size))
            data = torch.tensor(data)
            data = torch.unsqueeze(data,0)
            data = data.numpy()


        # Create measurement voltage data (training input data)
        measure_data = pd.read_csv(self.tra_path + '//' + item)
        measure_data = measure_data.to_numpy()
        measure_data = measure_data.T
        end_data = measure_data[1] - measure_data[0]
        end_data = self.normalization_data(end_data)
        end_data = np.reshape(end_data,(8,5))
        end_data = Fusion_splicing(end_data,64)
        end_data = torch.tensor(end_data)
        end_data = torch.unsqueeze(end_data,0)
        end_data = end_data.numpy()
        end_data = end_data.astype(np.float32)
        data = data.astype(np.float32)

        
        return end_data,data
    
    def normalization_data(self,data):
        data_max = data.max()
        data_min = data.min()
        gap = data_max - data_min
        temp_data = data - data_min
        temp_data = temp_data / gap
        return temp_data    
            


def Mydata(trg_dir,tra_dir,batch_size,image_size,category = 'circle'):
     
    dataset = MyDateset(trg_dir,tra_dir,image_size,category)
    loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

class Create_Gaussian_Template():

    def __init__(self, image_size, mask_r):
        self.image_size = image_size
        self.mask_r = mask_r
        self.Circle_mask = self.Creat_Circle_Mask()
        self.Circle_mask = torch.tensor(self.Circle_mask).to(dist_util.dev())
        self.Line_mask = self.Create_Line_Mask()
        self.Line_mask = torch.tensor(self.Line_mask).to(dist_util.dev())


    def distance_coord(self,x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def Creat_Circle_Mask(self):

        mask  = np.zeros(( self.image_size, self.image_size))   
        center = [self.image_size/2, self.image_size/2]
        for i in range( self.image_size):
            for j in range( self.image_size):
                if self.distance_coord(i,j,center[0],center[1]) < self.mask_r:
                    mask[i][j] = 1
        return mask
    
    def Create_Line_Mask(self):
        mask  = np.zeros(( self.image_size, self.image_size))   
        center = [self.image_size/2, self.image_size/2]
        for i in range( self.image_size):
            for j in range( self.image_size):
                if self.distance_coord(i,j,center[0],center[1]) < self.mask_r:
                    mask[i][j] = 1
        mask = mask.reshape((1,4096))
        return mask
    
    def Gaussian_Circle_Template(self, x_0):
        batch_size2 = x_0.shape[0]
        #print(batch_size2)
        
        noise = torch.randn_like(x_0)
        for i in range(batch_size2):
            noise[i] = noise[i] * self.Circle_mask

        return noise
    
    def Gaussian_Line_Template(self, x_0):
        batch_size2 = x_0.shape[0]
        noise = torch.randn_like(x_0)
        for i in range(batch_size2):
            noise[i] = noise[i] * self.Line_mask

        return noise


    def Deal_Mode_Out(self,mode_out):
        batch_size2 = mode_out.shape[0]
        for i in range(batch_size2):
            mode_out[i] = mode_out[i] * self.Circle_mask
        return mode_out

if __name__ == "__main__":
    path = "E://SWUST//LCW//WORK//WORK//EIT//open source//data//circle//coord//1000.csv"
    distribution_data = pd.read_csv(path)
    distribution_data = distribution_data.to_numpy()
    distribution_data = distribution_data.T
    datamodel = Create_Eit_DataModel(distribution_data,1,64,0.20)
    image = datamodel.Create_Eit_DataModel()
    data = np.reshape(image,(64,64))
    print(data.shape)
    plt.subplot(111)
    plt.imshow(data)
    plt.show()
    