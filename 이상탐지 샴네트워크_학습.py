# Import ======================================================================
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
from tqdm.notebook          import tqdm_notebook

%matplotlib inline
import pylab                as pl
from IPython                import display

from 이상탐지 샴네트워크_구조      import *
# Import ======================================================================


# Setting =====================================================================
loopTotNumber   = 10000
pbar_0          = tqdm_notebook(range(loopTotNumber), total=loopTotNumber)
tgtDf           = []
lossHist        = []
stopIdx         = -1
# Setting =====================================================================



# Train =======================================================================
batchInfoDf = pd.read_csv(batchInfoPath)
for idx in pbar_0:
    
    # Sampling 
    sampleIdx = np.random.randint(len(batchInfoDf), size=1)
    values = batchInfoDf.iloc[sampleIdx, :]
    
    # get batch image list
    imgList = pd.read_csv(values.batchImagePath)
    
    for i in range(1):
        
        img0, img1, label = [], [], []
        
        siamese_dataset = SiameseNetworkDataset(imgList         = imgList
                                                , tgtTnB        = args.TnB
                                                , tgtString     = args.String
                                                , removeString  = args.removeString
                                                , transform     = None
                                                , rx            = rx
                                                , ry            = ry
                                                , rc            = rc)
        
        siamese_dataset=iter(siamese_dataset)
        
        try:
            print('img-getter')
            for bat in range(batchSize):
                
                tempImg0, tempImg1, tempLabel = next(siamese_dataset)
                
                img0.append(tempImg0)
                img1.append(tempImg1)
                label.append(tempLabel)
        except:
            print('Error')
            print(traceback.format_exc())
            break
        else:
            print('Learn')
            img0    = tf.convert_to_tensor(np.array(img0))
            img1    = tf.convert_to_tensor(np.array(img1))
            label   = tf.convert_to_tensor(np.array(label))
            
            history = model.fit([img0, img1]
                                , label
                                , validation_data   = ([img0, img1, label])
                                , batch_size        = batchSize
                                , epoch             = 1
                                , verbose           = 0
                                , callbacks         = [mcp_save])
            
            lossHist.append(history.history['loss'][-1])
            
            if i%5==0:
                pl.plot(lossHist)
                pl.tile('')
                display.clear_output(wait=True)
                display.display(pl.gcf())
                plt.figure()
                plt.plot(lossHist, label='train_loss')
                #plt.savefig()
                plt.cla()
                plt.close()
# Train =======================================================================



# Valid =======================================================================
import tqdm
for imgPath in tqdm.tqdm(imgList):
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    img = SiameseNetworkDataset.imgPrepare(img=img, imgPath=imgPath, devYN=True, rx=rx, ry=ry)
    tgtDf.append(model.get_layer('model')([np.array([img])]).numpy()[0])
    
tgtDf = pd.DataFrame(tgtDf)
tgtDf['imgList']    = imgList
tgtDf['FLAG']       = ['BAD' if re.compile(args.tgtString).match(i) else 'GOOD' for i in tgtDf['imgList']]

plt.scatter(x =tgtDf.loc[tgtDf['FLAG'].isin(['GOOD']), '0']
            , y =tgtDf.loc[tgtDf['FLAG'].isin(['GOOD']), '1']
            , c ='b')
            
plt.scatter(x =tgtDf.loc[tgtDf['FLAG'].isin(['BAD']), '0']
            , y =tgtDf.loc[tgtDf['FLAG'].isin(['BAD']), '1']
            , c ='r')
plt.show()
# Valid =======================================================================