# -*- coding: utf-8 -*-

# BASE SETTING =====================================
import os
import traceback
import cv2
import matplotlib.pyplot    as plt
import numpy                as np
from skimage.util           import  vies_as_windows

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# BASE SETTING =====================================



# for DEEP-LEARNING ================================
import torchvision
import torchvision.datasets     as dset
import torchvision.transforms   as transforms
from  torch.utils.datasets      import DataLoader, Dataset

import tensorflow                   as tf
import tensorflow.keras.backend     as K
from tensorflow.keras.models        import Model
from tensorflow.keras.layers        import Input
from tensorflow.keras.layers        import Conv2D, Convd3D
from tensorflow.keras.layers        import Dense
from tensorflow.keras.layers        import Flatten
from tensorflow.keras.layers        import Dropout
from tensorflow.keras.layers        import BatchNormalization
from tensorflow.keras.layers        import GlobalAveragePooling2D, GlobalAveragePooling3D
from tensorflow.keras.layers        import MaxPooling2D, MaxPooling3D
from tensorflow.keras.layers        import Lambda, Add
from tensorflow.keras.layers        import LeakyReLU
from tensorflow.keras.optimizers    import Adam
from tensorflow.keras.callbacks     import ReduceLROnPlateau
from tensorflow.keras.callbacks     import ModelCheckpoint

tf.keras.backend.clear_session()
# for DEEP-LEARNING ================================



# init =============================================
rx, ry, rc              = 1600, 150, 3
inputX, inputY, inputC  = 1600, 70, 3
batchSize               = 32
loadExistModel          = True
embedDim                = 2
# init =============================================



# Create : Dataset Class ==========================
class SiameseNetworkDataset(Dataset):
    
    def __init__(self, imgList:list=None, tgtTnB:str=None, tgtString:str=None
                , removeString:str=None, transform:object=None
                , rx:int=1600, ry:int=150, rc:int=3):
        
        self.imgList        = imgList
        self.tgtTnB         = tgtTnB
        self.tgtString      = tgtString
        self.removeString   = removeString
        
        self.imgListSub     = [i for i in self.imgList if self.tgtTnB in i]
        
        if self.removeString is not None:
            self.imgList_0  = [i for i in self.imgListSub if (not re.compile(self.tgtString).match(i)) & (not re.compile(self.removeString).match(i))]
        else:
            self.imgList_0  = [i for i in self.imgListSub if (not re.compile(self.tgtString).match(i))]
        
        self.imgList_1      = [ i for i in self.imgListSub if re.compile(self.tgtString).match(i)]
        
        self.transform      = transform
        
        self.rx             = rx
        self.ry             = ry
        self.rc             = rc
        
    @staticmethod
    def imgPrepare(img:object=None, imgPath:str=None, ROI:object=None, devYN:bool=False, rx:int=1600, ry:int=150, rc:int=3):
        
        if devYN:
            alpha                   = 5
            
            imgOri                  = img.copy()
            img                     = cv2.resize(img, dsize=(0, 0), fx=1/alpha, fy=1/alpha, interpolation=cv2.INTER_LINEAR)
            
            roiDict                 = ROI(img, imgPath)
            minX, minY, maxX, maxY  = roiDict['TAPE']
            
            img                     = imgOri[alpha*minY:alpha*maxY, alpha*minX:alpha*maxX]
            
        img                         = cv2.resize(img, dsiz=(rx, ry), interpolation=cv2.INTER_LINEAR)
        
        img                         = (img/255).astype(np.float32)
        
        return(img)
    
    
    def __getitem__(self, index):
    
        difference_class = random.randint(0, 1)
        
        if difference_class:
            # Choise 0 and 1 : Good & Bad Class
            imgPath0    = random.choice(self.imgList_0)
            img0        = cv2.imread(imgPath0, cv2.IMREAD_COLOR)
            img0        = SiameseNetworkDataset.imgPrepare(img=img0, imgPath=imgPath0, devYN=True
                                                            , rx=self.rx, ry=self.ry, rc=self.rc)
                                                            
            imgPath1    = random.choice(self.imgList_1)
            img1        = cv2.imread(imgPath01, cv2.IMREAD_COLOR)
            img1        = SiameseNetworkDataset.imgPrepare(img=img1, imgPath=imgPath1, devYN=True
                                                            , rx=self.rx, ry=self.ry, rc=self.rc)
            
            return(img0, img1, np.array([1], dtype=np.float32))
        
        else:
            same_class = random.randint(0, 1)
            
            if same_class:
                # Choice 0 : Good Class
                imgPath0    = random.choice(self.imgList_0)
                img0        = cv2.imread(imgPath0, cv2.IMREAD_COLOR)
                img0        = SiameseNetworkDataset.imgPrepare(img=img0, imgPath=imgPath0, devYN=True
                                                                , rx=self.rx, ry=self.ry, rc=self.rc)
                                                                
                imgPath1    = random.choice(self.imgList_0)
                img1        = cv2.imread(imgPath01, cv2.IMREAD_COLOR)
                img1        = SiameseNetworkDataset.imgPrepare(img=img1, imgPath=imgPath1, devYN=True
                                                                , rx=self.rx, ry=self.ry, rc=self.rc)
            else:
                # Choice 1 : Bad Class
                imgPath0    = random.choice(self.imgList_1)
                img0        = cv2.imread(imgPath0, cv2.IMREAD_COLOR)
                img0        = SiameseNetworkDataset.imgPrepare(img=img0, imgPath=imgPath0, devYN=True
                                                                , rx=self.rx, ry=self.ry, rc=self.rc)
                                                                
                imgPath1    = random.choice(self.imgList_1)
                img1        = cv2.imread(imgPath01, cv2.IMREAD_COLOR)
                img1        = SiameseNetworkDataset.imgPrepare(img=img1, imgPath=imgPath1, devYN=True
                                                                , rx=self.rx, ry=self.ry, rc=self.rc)
                                                                
            return(img0, img1, np.array([0], dtype=np.float32))
            
    
    def __len__(self):
        return(len(self.imgList))
# Create : Dataset Class ==========================



# Create : Build Model Structure ==================
def build_siamese_model(inputShape:ojbect=None, embeddingDim:int=2):
        
        # inputs
        inputs      = Input(inputShape)
        x_shortcut  = inputs
        
        # conv 1
        x           = Conv2D(4, (7, 7), padding='same')(inputs)
        x           = BatchNormalization()(x)
        x           = LeakyReLU()(x)
        x           = MaxPooling2D(pool_size=(7, 7), strides=(2, 2), padding='same')(x)
        
        # conv 2
        x           = Conv2D(8, (7, 7), padding='same')(inputs)
        x           = BatchNormalization()(x)
        x           = LeakyReLU()(x)
        x           = MaxPooling2D(pool_size=(7, 7), strides=(2, 2), padding='same')(x)
        
        # conv 3
        x           = Conv2D(16, (7, 7), padding='same')(inputs)
        x           = BatchNormalization()(x)
        x           = LeakyReLU()(x)
        x           = MaxPooling2D(pool_size=(7, 7), strides=(2, 2), padding='same')(x)
        
        # network
        flattenedOutput = Flatten()(x)
        
        denseOutputs    = Dense(32)(flattenedOutput)
        denseOutputs    = BatchNormalization()(denseOutputs)
        
        denseOutputs    = Dense(16)(flattenedOutput)
        denseOutputs    = BatchNormalization()(denseOutputs)
        
        outpus          = Dense(embeddingDim, activation='sigmoid')(denseOutputs)
        
        # build model
        model           = Model(inputs, outpus)
        
        return(model)
        
# Create : Build Model Structure ==================



# Create : Define Loss and Metric Function ========
def euclidean_distance(vectors:object=None):
    
    (featsA, featsB)    = vectors
    
    # compute sum of square between tow image feature vectors
    sumSquared          = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    
    return(K.sqrt(K.maximum(sumSquared, K.epsilon())))


def contrastive_loss_with_margin(margin:float=0):
    
    def contrastive_loss(y_true:float=0, y_pred:float=0):
        
        square_pred     = tf.math.square(y_pred)
        
        margin_square   = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    
        return(tf.math.reduce_mean((1-ytrue)*square_pred + y_true*margin_square))
        
    return(contrastive_loss)
# Create : Define Loss and Metric Function ========



if __name__ == '__main__':
    
    imgA                = input(shape=(inputY, inputX, inputC))
    imgB                = input(shape=(inputY, inputX, inputC))
    
    # CNN Model
    featureExtractor    = build_siamese_model((inputY, inputX, inputC), embedDim)
    featsA              = featureExtractor(imgA)
    featsB              = featureExtractor(imgAB)
    
    # finally construct the siamese network
    outpus              = Lambda(euclidean_distance)([featsA,  featsB])
    model               = Model(inputs=[imgA, imgB], outputs=outputs, name='MODEL_001')
    
    if loadExistModel:
        model.load_weights(r'')
    
    # compile
    adamBack    = tf.keras.optimizers.Adam(learning_rate=0.001/1)
    mcp         = ModelCheckpoint(r'', save_best_only = True, monitors='loss', mode='min')
    reduce_lr   = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5)
    
    model.compile(loss=contrastive_loss_with_margin(margin=1.5*0.5*((embedDim)**(1/2))), optimizer=adamBack)
