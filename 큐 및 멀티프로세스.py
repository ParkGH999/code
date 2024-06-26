# Import ======================================================================
from multiprocessing import Pool, Process, Queue, Manager
# Import ======================================================================



# Init ========================================================================
toDLQueue = Queue()
resultListML = manager().list()
resultListDL = manager().list()
# Init ========================================================================



# Run Process =================================================================
ProcessList = []
for i in range(args.mpCore):
	subImgList = [imgPath for idx, imgPath in enumerate(img_list) if idx%args%mpcore == i]
	proc = Process(target=MAIN_IP_DP(
                                    imgList = subImgList, inputImageQueue=None, toDLQueue=toDLQueue,
                                    pbar = tqdm(subImgList, total=len(subImgList), mininterval=5),
                                    mlProcessNum=args.mpCore, ROI_MOD_NM=ROI_MOD_NM, VISION_NAME=VISION_NAME,
                                    verbose=1).run,
                    args=(resultListML, ),
                    daemon=True)
	ProcessList.append(proc)
	proc.start()


proc = Process(target = MAIN_SP_DL_TEST(
                                        imgCount=len(imgLsit),
                                        toDLQueue=toDLQueue,
                                        modelPath='',
                                        pbar = tqdm(imgLsit, total=len(imgLsit), mininterval=5),
                                        mlProcessNum=args.mpCore, ROI_MOD_NM=ROI_MOD_NM, VISION_NAME=VISION_NAME,
                                        verbose=1).run,
                args=(resultListDL, ),
                daemon=True)
ProcessList.append(proc)
proc.start()



for proc in ProcessList:
proc.join()
# Run Process =================================================================



# Get Result ==================================================================
resultListML = list(resultListML)[0]
resultListDL = list(resultListDL)[0]
# Gert Result =================================================================