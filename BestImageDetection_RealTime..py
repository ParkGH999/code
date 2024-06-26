# -*- coding: utf-8 -*-

"""
Description 
- ???
- ???
Author
- ???
Latest updates
- ???
"""

# Module setting ==============================================================
try:
    print("Start : Import Modules")
    
    # Default
    import sys
    import os
    import traceback
    import threading
    import datetime
    import socket
    import io
    import mmap
    import cv2
    import ctypes
    import pandas   as pd
    import numpy    as np
    from os         import path
    pd.set_option('display.max_columns', 20)
    
    print(os.getpid())
    SPAPath = "E:\\[01]TEST\\SPA"
    sys.argv.append("E:\\[01]TEST\\SPA\\Parents.json")
    sys.argv.append("MARIA")
    sys.argv.append('GPU')
    sys.argv.append('GPU-d6fec61c-46b7-9033-2a77-3e1f0a7f6515')
    
    # Image
    from keras.preprocessing    import image
    from sklearn                import preprocessing
    
    # Model
    import tensorflow           as tf
    import json
    import keras.backend.tensorflow_backend as KTF
    
    from keras              import backend as K
    from keras.models       import model_from_json, Model
    from sklearn.metrics    import confusion_matrix
    
    sys.path.append(SPAPath)
    sys.path.append(os.path.join(SPAPath, "CNN_MODEL_NETWORK"))
    
    try:
        ctypes.windll.kernel32.SetConsoleTitleW(os.path.basename(__file__).replace(".py",""))
    except:
        OrigStdout = sys.stdout
        AnalLog = open(sys.argv[1].replace("json", "txt"), 'w')
        sys.stdout = AnalLog    
    print("Fnish : Import Modules")
except:
    print("Error : Import Modules", sys.exc_info())
    print(traceback.format_exc())
    sys.exit(2)
# Module setting ==============================================================



from TEST import logging_setting, report_prgs_stat, property_setting
from TEST import data_import, data_update, data_export
from TEST import findfvlist
from TEST import insert_key_to_lern_fv_table, join_deft_rank
from TEST import timestamp, DbTimeFun
from TEST import standardize, timesubset, save_model_file
from TEST import JobID, SimID, SimVer, NodeID
from TEST import FctID, ModlID, OperID, ProdID
from TEST import ExeType, JobType, NodeType, ExeServerID
from TEST import SimlDeftTypeCode, load_image, load_model_file

from IndexSeq import nodenum

# Load All Parameters =========================================================
def load_all_parameter(FilePath=sys.argv[1]):  
    #------------------------------------------------------
    # Description
    # - Load Json File
    # Arguments
    # - FilePath : Json File Path And Name
    #------------------------------------------------------
    
    Parameters = json.loads(open(FilePath, encoding='UTF8').read())
    return(Parameters)

# Run Function
Parameters = load_all_parameter()
# Load All Parameters =========================================================


# DataBase Connetion Info =====================================================
# Get Import Statement And DataBase Connetion Information
import ParameterDB
DB = ParameterDB.DBinfo(sys.argv[2])
DBimport, DbConnectionInfo, DbConnectionUrl = DB.DbConInfo()
exec(DBimport)
# DataBase Connetion Info =====================================================


# Basic Setting ===============================================================
try:
    print("Start : Global Parameter")
    
    TableKey = ["JOB_ID", "RLTM_HNDL_OCUR_ID", "DEFT_ID",
                "IMG_ID", "DEFT_SEQ", "IMG_TYPE_SEQ"]
    BasicCol = ["NOISE_DIV", "LAST_DEFT_TYPE_CODE",
                "LAST_DTL_DEFT_TYPE_CODE", "INSP_STD_DTM"]
    
    print("Fnish : Global Parameter")
except:
    print("Error : Global Parameter", sys.exc_info())
    print(traceback.format_exc())
    sys.exit(3)
# Basic Setting ===============================================================


# Define Function==============================================================
print("Start : Define Function")

def img2arr(FILE_PATH, color_mode="grayscale", size=(80, 80), MeomoryInput="", file_size=""):
    
    ncol, nrow = size
    noimg = []
    
    if color_mode=="grayscale":
        img_arr = np.empty((len(FILE_PATH), ncol, nrow, 1), dtype=np.uint8)
    else:
        img_arr = np.empty((len(FILE_PATH), ncol, nrow, 3), dtype=np.uint8)
    
    for idx, img_path in enumerate(FILE_PATH):
        try:
            print("ReadImg : ",datetime.datetime.now())
            if "File" in MeomoryInput:
                img = cv2.imread(img_path)
            else:
                shmem1 = mmap.mmap(0, file_size, img_path, mmap.ACCESS_READ)
                msg_bytes = shmem1.read()
                Final_bytes = msg_bytes
                img = cv2.imdecode(np.fromstring(Final_bytes, dtype='uint8'), 1)
            print("ReadImg : ",datetime.datetime.now())
                
            if color_mode=="grayscale":
                print("Color : ",datetime.datetime.now())
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print("Color : ",datetime.datetime.now())
            print("Resize : ",datetime.datetime.now())
            img = cv2.resize(img, (ncol, nrow))
            print("Resize : ",datetime.datetime.now())
        except:  # path가 없는 경우에는 pass!
            noimg.append(idx)
        
        print("Assign : ",datetime.datetime.now())
        if color_mode=="grayscale":
            img_arr[idx,:,:,0] = img
        else:
            img_arr[idx,:,:,:] = img
        print("Assign : ",datetime.datetime.now())
    return img_arr


def ImgPreprocessing(X_Data):
    print("I : ",datetime.datetime.now())
    X_Data  = X_Data.astype("float32")
    X_Data   -= 128. 
    X_Data   /= 128.
    print("J : ",datetime.datetime.now())
    return X_Data
# Using Data Preprocessing==============================


# Model Load ===========================================
def load_cnn_model(ToPath, json_path, bestmodel_path):
    whole_json = open(path.join(ToPath, json_path)).read()
    model_json = whole_json[:whole_json.find(', "labels"')]+'}'
    # input image size
    model_infor   = json.loads(model_json)
    layer_config  = model_infor["config"]["layers"]
    input_imgsize = tuple(layer_config[0]['config']['batch_input_shape'][1:3])
    # model
    trained_model = model_from_json(model_json)
    # Labels
    label_start = whole_json.find('"labels": "')+len('"labels": "')
    LabelVals   = np.sort(list(eval(whole_json[label_start:-2])))
        
    trained_model.load_weights(path.join(ToPath, bestmodel_path))
    return trained_model, LabelVals, input_imgsize
# Model Load ===========================================


# Thread 1 : predict ===================================
class Run_Gpu_Thread:
    def __init__(self, tr_model, tr_LabelVals):
        self.tr_model = tr_model
        self.tr_LabelVals = tr_LabelVals
        self.tr_model._make_predict_function()
        self.session = K.get_session()
        self.graph = tf.get_default_graph()
        self.graph.finalize()
        self.result = pd.DataFrame()
    
    def CnnClassifier(self, X_test, SKIP_FLAG, FILE_NM, ImgDataJsonStr):
        # model predict
        with self.session.as_default():
            with self.graph.as_default():
                print("Predict : ",datetime.datetime.now())
                pred            = self.tr_model.predict(X_test)
                print(pred)
                print("Predict : ",datetime.datetime.now())
                
                print("Post-Process : ",datetime.datetime.now())
                ProbSortLabels  = tr_LabelVals[pred.argmax()]
                test_DF = {"LAST_PRDCT_DEFT_TYPE_CODE" : ProbSortLabels,
                           "MAX_PROB" : pred.max()}
                test_DF.update({"SKIP_FLAG" : SKIP_FLAG, "IMG_NM":FILE_NM,
                                "JSON":ImgDataJsonStr, "PRED_TIME":datetime.datetime.now()})
                print(ProbSortLabels)
                print("Post-Process : ",datetime.datetime.now())
                return(test_DF)
print("Fnish : Define Function")
# Thread 1 : predict ===================================
# Define Function==============================================================



# Assign Property Value To Object =============================================
try:
    print("Start : Parameter Setting")
    
    UseGPU = sys.argv[3]
    assert UseGPU in ["GPU", "CPU"]
    UseGPU = True if UseGPU=="GPU" else False
    
    color_mode    = "rgb" # ["grayscale", "rgb"]
    
    print("Fnish : Parameter Setting")
#    print(NodeInfo)
except:
    print("Error : Parameter Setting", sys.exc_info())
    print(traceback.format_exc())
    sys.exit(5)
# Assign Property Value To Object =============================================


# Logger Setting ==============================================================
MyLogger=""
#try:
#    print("Start : Logger Setting")
#    
#    MyLogger = logging_setting(EndDTM="0000-00-00")
#    
#    report_prgs_stat(CDKey="000000", StatCD="S",
#                     AdditionalStatement="Start : BestImageDetection",
#                     BaseTM="", total="1", now="0",
#                     MyLogger=MyLogger,
#                     DbConnectionInfo=DbConnectionInfo)
#    
#    report_prgs_stat(CDKey="000000", StatCD="R",
#                     AdditionalStatement=
#                     ", ".join([str(i) for i in NodeInfo]).replace("'", ""),
#                     BaseTM="", total="1", now="0",
#                     MyLogger=MyLogger, DbConnectionInfo=DbConnectionInfo)
#    
#    print("Fnish : Logger Setting")
#except:
#    print("Error : Logger Setting", sys.exc_info())
#    print(traceback.format_exc())
#    sys.exit(6)
# Logger Setting ==============================================================


# Using GPU setting ===========================================================
# GPU 설정 초기화
try:
    print("Start : GPU Setting")
#    report_prgs_stat(CDKey="000000", StatCD="R",
#                     AdditionalStatement="Start : GPU Setting",
#                     BaseTM="", total="1", now="0",
#                     MyLogger=MyLogger, DbConnectionInfo=DbConnectionInfo)
    
    sys.path.append(os.path.join(SPAPath, "SetGPU"))
    import GPU_setting
    import GPUtil
    
    if UseGPU: ## Using GPU
        GPUInfo = GPUtil.getGPUs()
        for GPU_ID, SubGPU in enumerate(GPUInfo):
            if SubGPU.uuid == sys.argv[4]:
                break
        
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"   
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
        KTF.set_session(GPU_setting.get_session(gpu_num=GPU_ID)) # , gpu_frac=0.3
    else:
        KTF.set_session(GPU_setting.get_session(gpu_use=False))
    print("Finish : GPU Setting")
#    report_prgs_stat(CDKey="000000", StatCD="R",
#                     AdditionalStatement="Finish : GPU Setting",
#                     BaseTM="", total="1", now="0",
#                     MyLogger=MyLogger, DbConnectionInfo=DbConnectionInfo)
except:
    print("Error : GPU Setting", sys.exc_info())
    print(traceback.format_exc())
#    report_prgs_stat(CDKey="000000", StatCD="E",
#                     AdditionalStatement="Error : GPU Setting",
#                     BaseTM="", total="1", now="0",
#                     MyLogger=MyLogger, DbConnectionInfo=DbConnectionInfo)
    sys.exit(7)
# Using GPU setting ===========================================================


# Load Model Info =============================================================
try:
    print("Start : Load Model Info")
#    report_prgs_stat(CDKey="000000", StatCD="R",
#                     AdditionalStatement="Start : Load Model Info",
#                     BaseTM="", total="1", now="0",
#                     MyLogger=MyLogger, DbConnectionInfo=DbConnectionInfo)
#    
#    ModelInfo = load_model_file(ModelModelID=ModelModelID,
#                                ModelSimulID=ModelSimulID, 
#                                ModelSimulVer=ModelSimulVer,
#                                ModelNodeID=ModelNodeID,
#                                RefTPJobID=RefTPJobID, 
#                                Model=ModelName,
#                                ModelOut=False,
#                                DbConnectionInfo=DbConnectionInfo)
    ModelPath = "E:\\TEST\\SP\\SEAH.BESTEEL\\SMALL.ROLLING\\ACM1" #ModelInfo.ix[0, "LERN_RSLT_FILE_ROUT"]
    
    DirList = [os.path.join(ModelPath, i) for i in os.listdir(ModelPath)]
    
    ModelFile = [i for i in DirList if ".hdf5" in i][0]
    ModelJson = [i for i in DirList if ".json" in i][0]
    #RegionSavePath = ToPath +"\\Region"
    
    # Load trained cnn model
    tr_model, tr_LabelVals, input_imgsize = load_cnn_model(ModelPath, ModelJson, ModelFile)
    
    Best_Img_Cut = Run_Gpu_Thread(tr_model, tr_LabelVals)
    
    print("Finish : Load Model Info")
#    report_prgs_stat(CDKey="000000", StatCD="R",
#                     AdditionalStatement="Finish : load img file path and defect type",
#                     BaseTM="", total="1", now="0",
#                     MyLogger=MyLogger, DbConnectionInfo=DbConnectionInfo)
except:
    print("Error : Load Model Info", sys.exc_info())
    print(traceback.format_exc())
#    report_prgs_stat(CDKey="000000", StatCD="E",
#                     AdditionalStatement="Error : Load Model Info",
#                     BaseTM="", total="1", now="0",
#                     MyLogger=MyLogger, DbConnectionInfo=DbConnectionInfo)
    sys.exit(9)
# Load Model Info =============================================================
   

#==============================================================================
print("Setting Socket and Queue")

TO_HOST = '127.0.0.1'
TO_PORT=14201

HOST='127.0.0.1' #호스트를 지정하지 않으면 가능한 모든 인터페이스를 의미한다.
PORT=8009 #포트지정
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST,PORT))
s.listen(1)#접속이 있을때까지 기다림

from queue import Queue
JsonQue = Queue()
RsltQue = Queue()


class producer(threading.Thread):
    def run(self):
        while True:
            conn, addr=s.accept()
            ImgDataJson =conn.recv(1024)
            conn.send(b'ReceiveOk')
            if "CLASSIFICATION" not in ImgDataJson.decode():
                conn.send(b'Running')
                continue
            JsonQue.put(ImgDataJson)
            print(list(JsonQue.queue))
# JsonQue.put(b'{"CAM_ID":"CAM1","CAM_NAME":"SEA_VIDEO","FILE_TYPE":"File","FILE_PATH":"D:\\CAM1\\image-00230.png","FILE_SIZE":0,"SKIP_FLAG":"F","COMMAND":"CLASSIFICATION","JOB_ID":null,"EXE_TYPE":"RT"}')

class consumer(threading.Thread):
    def run(self):
        while True:
            if not JsonQue.empty():
                print("Working Thread is ", threading.currentThread().getName())
                print(threading.get_ident())
                # Load img and defect label and Data Preprocessing ====================
                try:
                    StartTime = datetime.datetime.now()
                    print("\n\n\nStart : {}".format(str(StartTime))+"\n\n\n")
                    ImgDataJsonOriginal = JsonQue.get()
                    if isinstance(ImgDataJsonOriginal, bytes):
                        ImgDataJson = ImgDataJsonOriginal.decode()
                    ImgDataJsonStr = ImgDataJson.replace("\\", "\\\\")
                    ImgDataJsonStr = ImgDataJsonStr.replace("null", '""')
                    ImgDataJson = eval(ImgDataJsonStr)
                    print("JsonInfo : ",ImgDataJson)
                    print("Start IP: ",datetime.datetime.now())
                    X_test = img2arr(FILE_PATH = [ImgDataJson["FILE_PATH"]],
                                     color_mode=color_mode,
                                     size=input_imgsize,
                                     MeomoryInput= ImgDataJson["FILE_TYPE"],
                                     file_size=ImgDataJson["FILE_SIZE"])
                    X_test  = X_test.astype("float32")
                    print("End IP : ",datetime.datetime.now())
                    print("ImageProcess Time is ", datetime.datetime.now() - StartTime)
                except:
                    sys.exit(1000)
                # Load img and defect label and Data Preprocessing ================
                
                # Classification===================================================
                try:
                    StartTime = datetime.datetime.now()
                    test_DF = Best_Img_Cut.CnnClassifier(X_test, ImgDataJson["SKIP_FLAG"],
                                                         ImgDataJson["FILE_PATH"],
                                                         ImgDataJsonStr)
                    RsltQue.put(test_DF)
                    print("Working Time is ", datetime.datetime.now() - StartTime)
                except:
                    sys.exit(1100)
#==============================================================================


#==============================================================================
print("Start Process")
consumer().start()
producer().start()

Rslt_Before = {"SKIP_FLAG": "N", "JSON" : "{}", "LAST_PRDCT_DEFT_TYPE_CODE":""}
Rslt_Now = {"SKIP_FLAG": "N", "JSON" : "{}", "LAST_PRDCT_DEFT_TYPE_CODE":""}
WorkingYN = False
SendYN = False
while True:
    try:
        CurrentTime = datetime.datetime.now()
        
        if not RsltQue.empty():
            # Detect Best Cut======================================            
            Rslt_Before = Rslt_Now
            Rslt_Now = RsltQue.get()
            
            if all([Rslt_Before["SKIP_FLAG"] == "T", Rslt_Now["SKIP_FLAG"] == "F"]):
                Rslt_Before = {"SKIP_FLAG": "N", "JSON" : "{}", "LAST_PRDCT_DEFT_TYPE_CODE":""}
                SendYN = False
                
            if not SendYN:
                if all([Rslt_Before["SKIP_FLAG"] == "F", Rslt_Now["SKIP_FLAG"] == "T"]):
                    
                    print("\n=======================Case 2=======================\n")
                    for i in list(zip([Rslt_Before, Rslt_Now], ["F", "F"])):
                        if all([isinstance(i[0]["JSON"], str), i[0]["SKIP_FLAG"] != "N"]):
                            MSG = i[0]["JSON"]
                            if i[1] == "T":
                                MSG = MSG.replace('"LAST_PRDCT_DEFT_TYPE_CODE":""',
                                                  '"LAST_PRDCT_DEFT_TYPE_CODE":"'+i[0]["LAST_PRDCT_DEFT_TYPE_CODE"]+'"')
                                MSG = MSG.replace('"DEFT_TYPE_CODE_1":""',
                                                  '"DEFT_TYPE_CODE_1":"'+i[0]["LAST_PRDCT_DEFT_TYPE_CODE"]+'"')
                                MSG = MSG.replace('"REPR_DEFT_YN":""', '"REPR_DEFT_YN":"Y"')
                                MSG = MSG.replace('"DEFT_TYPE_CODE_PRBL_1":""',
                                                  '"DEFT_TYPE_CODE_PRBL_1":"'+str(i[0]["MAX_PROB"])+'"')
                            MSG = MSG.replace("}", ', "BestYN": "' + i[1] + '"}')
                            c=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                            c.connect((TO_HOST, TO_PORT))
                            c.send(eval("b'" + MSG + "'"))
                            c.close()
                            print("\nCase 2 : ", datetime.datetime.now()-CurrentTime)
                            print(MSG)
                    SendYN = True
                    
                elif all([Rslt_Now["LAST_PRDCT_DEFT_TYPE_CODE"] == "SPARK", Rslt_Now["LAST_PRDCT_DEFT_TYPE_CODE"] == "OTHER"]):
                    print("\n=======================Case 3=======================\n")
                    for i in list(zip([Rslt_Before, Rslt_Now], ["T", "F"])):
                        if isinstance(i[0]["JSON"], str):
                            MSG = i[0]["JSON"]
                            if i[1] == "T":
                                MSG = MSG.replace('"LAST_PRDCT_DEFT_TYPE_CODE":""',
                                                  '"LAST_PRDCT_DEFT_TYPE_CODE":"'+i[0]["LAST_PRDCT_DEFT_TYPE_CODE"]+'"')
                                MSG = MSG.replace('"DEFT_TYPE_CODE_1":""',
                                                  '"DEFT_TYPE_CODE_1":"'+i[0]["LAST_PRDCT_DEFT_TYPE_CODE"]+'"')
                                MSG = MSG.replace('"REPR_DEFT_YN":""', '"REPR_DEFT_YN":"Y"')
                                MSG = MSG.replace('"DEFT_TYPE_CODE_PRBL_1":""',
                                                  '"DEFT_TYPE_CODE_PRBL_1":"'+str(i[0]["MAX_PROB"])+'"')
                            MSG = MSG.replace("}", ', "BestYN": "' + i[1] + '"}')
                            c=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                            c.connect((TO_HOST, TO_PORT))
                            c.send(eval("b'" + MSG + "'"))
                            c.close()
                            print("\nCase 3 : ", datetime.datetime.now()-CurrentTime)
                            print(MSG)
                    SendYN = True

                elif Rslt_Now["LAST_PRDCT_DEFT_TYPE_CODE"] == "OK":
                    
                    print("\n=======================Case 4=======================\n")
                    for i in list(zip([Rslt_Before, Rslt_Now], ["F", "T"])):
                        if all([isinstance(i[0]["JSON"], str), i[0]["SKIP_FLAG"] != "N"]):
                            MSG = i[0]["JSON"]
                            if i[1] == "T":
                                MSG = MSG.replace('"LAST_PRDCT_DEFT_TYPE_CODE":""',
                                                  '"LAST_PRDCT_DEFT_TYPE_CODE":"'+i[0]["LAST_PRDCT_DEFT_TYPE_CODE"]+'"')
                                MSG = MSG.replace('"DEFT_TYPE_CODE_1":""',
                                                  '"DEFT_TYPE_CODE_1":"'+i[0]["LAST_PRDCT_DEFT_TYPE_CODE"]+'"')
                                MSG = MSG.replace('"REPR_DEFT_YN":""', '"REPR_DEFT_YN":"Y"')
                                MSG = MSG.replace('"DEFT_TYPE_CODE_PRBL_1":""',
                                                  '"DEFT_TYPE_CODE_PRBL_1":"'+str(i[0]["MAX_PROB"])+'"')
                            MSG = MSG.replace("}", ', "BestYN": "' + i[1] + '"}')
                            c=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                            c.connect((TO_HOST, TO_PORT))
                            c.send(eval("b'" + MSG + "'"))
                            c.close()
                            print("\nCase 4 : ", datetime.datetime.now()-CurrentTime)
                            print(MSG)
                    SendYN = True
                
                elif Rslt_Now["LAST_PRDCT_DEFT_TYPE_CODE"]  in ["FINISH"]:
                    
                    print("\n=======================Case 5=======================\n")
                    for i in list(zip([Rslt_Before, Rslt_Now], ["T", "F"])):
                        if all([isinstance(i[0]["JSON"], str), i[0]["SKIP_FLAG"] != "N"]):
                            MSG = i[0]["JSON"]
                            if i[1] == "T":
                                MSG = MSG.replace('"LAST_PRDCT_DEFT_TYPE_CODE":""',
                                                  '"LAST_PRDCT_DEFT_TYPE_CODE":"'+i[0]["LAST_PRDCT_DEFT_TYPE_CODE"]+'"')
                                MSG = MSG.replace('"DEFT_TYPE_CODE_1":""',
                                                  '"DEFT_TYPE_CODE_1":"'+i[0]["LAST_PRDCT_DEFT_TYPE_CODE"]+'"')
                                MSG = MSG.replace('"REPR_DEFT_YN":""', '"REPR_DEFT_YN":"Y"')
                                MSG = MSG.replace('"DEFT_TYPE_CODE_PRBL_1":""',
                                                  '"DEFT_TYPE_CODE_PRBL_1":"'+str(i[0]["MAX_PROB"])+'"')
                            MSG = MSG.replace("}", ', "BestYN": "' + i[1] + '"}')
                            c=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                            c.connect((TO_HOST, TO_PORT))
                            c.send(eval("b'" + MSG + "'"))
                            c.close()
                            print("\nCase 5 : ", datetime.datetime.now()-CurrentTime)
                            print(MSG)
                    SendYN = True
                else:
                    if Rslt_Before["SKIP_FLAG"] == "F":
                        MSG = Rslt_Before["JSON"]
                        MSG = MSG.replace("}", ', "BestYN": "F"}')
                        MSG = "b'" + MSG + "'"
                        c=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                        c.connect((TO_HOST, TO_PORT))
                        c.send(eval(MSG))
                        c.close()
                        print("\nCase 6 : ", datetime.datetime.now()-CurrentTime)
                        print(MSG)
            else:
                if Rslt_Now["SKIP_FLAG"] != "N":
                    MSG = Rslt_Now["JSON"] 
                    MSG = MSG.replace("}", ', "BestYN": "F"}')
                    c=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                    c.connect((TO_HOST, TO_PORT))
                    c.send(eval("b'" + MSG + "'"))
                    c.close()
                    print("\nCase 1 : ", datetime.datetime.now()-CurrentTime)
                    print(MSG)
            # Send Best Cut========================================
    except:
        print(traceback.format_exc())
#sys.stdout = OrigStdout
#AnalLog.close()
#==============================================================================
