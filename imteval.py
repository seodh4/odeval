# from re import I
# from shutil import register_unpack_format
import sys
from PyQt5 import uic
from matplotlib.font_manager import json_dump
from eval import *
import numpy as np
import cv2
from datetime import datetime
import pandas as pd

import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns


from PyQt5.QtCore import pyqtSlot, Qt 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtGui

from tqdm import tqdm


from PyQt5.QtCore import QThread, pyqtSignal
import time

class External(QThread):
    """
    Runs a counter thread.
    """
    countChanged = pyqtSignal(int,int)
    outevalresult = pyqtSignal(dict,list)

    def __init__(self, parent): #parent는 WndowClass에서 전달하는 self이다.(WidnowClass의 인스턴스) 
        super().__init__(parent) 
        self.parent = parent #self.parent를 사용하여 WindowClass 위젯을 제어할 수 있다.
       

    def run(self):
        self.eval()
        # count = 0
        # while count < 100:
        #     count +=1
        #     time.sleep(1)
        #     self.countChanged.emit(count)




    def eval(self):

        iou_th = self.parent.doubleSpinBox_iou.value()
        
        iou_th=round(iou_th,3)
        try:
            gt_path = self.parent.gt_path
            det_path = self.parent.det_path
        except:
            QMessageBox.information(self,'오류','경로를 지정하시오')
            return


        total_gt_list = []
        total_over_gt_list = []
        total_e2e_gt_list = []
        total_fn_list = []

        total_tp_list = []
        total_e2e_tp_list = []
        total_fp_list = []
        total_overlab_list = []
        total_e2e_overlab_list = []

        class_label= []

        data = {}

        fileio = Fileio()
        eval = Eval()

        gtbox_list = []
        detbox_list = []

        # try:
        gtfile_list =os.listdir(self.parent.gt_path)
        gtfile_list = [file for file in gtfile_list if file.endswith(".json")]
        for idx, gtfile in enumerate(gtfile_list):
            detfile = gtfile
            gtbox_list = fileio.read_file(self.parent.gt_path+'/'+ gtfile, gtfile)
            detbox_list = fileio.read_file(self.parent.det_path+'/'+ detfile, gtfile)

            resultbox , score, label= eval.evaluation(gtbox_list, detbox_list, iou_th)
            data[gtfile] = {
                "resultbox" : resultbox, 
                "eval_score" : score
            }

            class_label += label
            total_tp_list += resultbox['tp_box']
            total_e2e_tp_list += resultbox['e2e_tp_box']
            total_fp_list +=resultbox['fp_box']
            total_fn_list += resultbox['fn_box']
            total_gt_list += resultbox['gt_box']
            total_over_gt_list += resultbox['over_gt_box']
            total_e2e_gt_list += resultbox['e2e_gt_box']
            total_overlab_list += resultbox['overlab_box']
            total_e2e_overlab_list += resultbox['e2e_overlab_box']

            self.countChanged.emit(idx,len(gtfile_list))

        # except:
        #     QMessageBox.information(self,'오류','파일 오류')
        #     return


        class_label_set = set(class_label) #집합set으로 변환
        class_label = list(class_label_set)

        total_det_n = len(total_tp_list)+ len(total_e2e_tp_list) + len(total_overlab_list) + len(total_e2e_overlab_list) + len(total_fp_list)
        total_gt_n = len(total_fn_list) + len(total_gt_list) + len(total_e2e_gt_list) + len(total_over_gt_list)

        total_det_precision =  round((len(total_tp_list + total_e2e_tp_list) / total_det_n),4)
        total_det_recall = round((len(total_tp_list + total_e2e_tp_list) / total_gt_n),4)
        total_e2e_precision =  round((len(total_e2e_tp_list) / total_det_n),4)
        total_e2e_recall = round((len(total_e2e_tp_list) / total_gt_n),4)

        
        det_Hmean = round(2 * (total_det_precision * total_det_recall) / (total_det_precision + total_det_recall),4)
        e2e_Hmean = round(2 * (total_e2e_precision * total_e2e_recall) / (total_e2e_precision + total_e2e_recall),4)


        result_data = {}
        now = datetime.now()
        result_data["project_name"] = self.parent.lineEdit_evalname.text()
        result_data["date"] = str(now)
        result_data["class_label"] = class_label
        result_data["IOU"] = iou_th
        result_data["result"] = data
        result_data["summary"] = {
            "total_det_precision" : total_det_precision,
            "total_det_recall": total_det_recall,
            "det_Hmean" : det_Hmean,
            "total_e2e_precision" : total_e2e_precision,
            "total_e2e_recall": total_e2e_recall,
            "e2e_Hmean" : e2e_Hmean
        }

        filename = self.parent.lineEdit_evalname.text() + '.json'
        json_dump(result_data,filename)
        # self.parent.print_result(result_data, class_label)
        self.parent.pushButton_evaluation.setEnabled(True)
        self.outevalresult.emit(result_data,class_label)
        self.parent.progressBar.setValue(0)



form_class = uic.loadUiType("windows.ui")[0]

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        self.pushButton_evaluation.clicked.connect(self.pushButton_evaluation_fuction)
        self.listWidget_filelist.itemDoubleClicked.connect(self.listWidget_filelist_DoubleClicked)
        self.tableWidget_filelist.doubleClicked.connect(self.tableWidget_filelist_doubleClicked)
        self.tableWidget_gt.doubleClicked.connect(self.tableWidget_gt_doubleClicked)

        self.toolButton_gtpath.clicked.connect(self.open_gtpath)
        self.toolButton_detpath.clicked.connect(self.open_detpath)
        self.toolButton_resultfile.clicked.connect(self.open_resultfile)
        self.pushButton_operresultfile.clicked.connect(self.oper_resultfile)
        
        self.gt_path = './gt/'
        self.det_path = './result/'



    # def onCountChanged(self, value):
    #     self.progressBar.setValue(value)


    def open_gtpath(self):
        self.gt_path=QFileDialog.getExistingDirectory(self,"Choose GT Directory","./")
        self.lineEdit_gtpath.setText(self.gt_path)
                    # QFileDialog.getOpenFileName(self,"Choose gtFile","./")
    def open_detpath(self):
        self.det_path=QFileDialog.getExistingDirectory(self,"Choose prediction Directory","./")
        self.lineEdit_detpath.setText(self.det_path)

    def open_resultfile(self):
        self.resultfile = QFileDialog.getOpenFileName(self,"Choose Result File","./")
        self.lineEdit_resultfile.setText(self.resultfile[0])
    
    def oper_resultfile(self):

        try:
            with open(self.resultfile[0], 'r') as f:
                result_data = json.load(f)
        except:
            QMessageBox.information(self,'오류','파일오류')
            return

        class_label = result_data['class_label']
        self.print_result(result_data, class_label)


    def tableWidget_gt_doubleClicked(self):
        

        row = self.tableWidget_gt.currentIndex().row()
        column = self.tableWidget_gt.currentIndex().column()
       

        file=self.tableWidget_gt.item(row, 3).text()
        gt_label=self.tableWidget_gt.item(row, 2).text()
        
        cordinate=[]
        cordinate_text=self.tableWidget_gt.item(row, 4).text()
        cordinate_text= cordinate_text.strip("]""[")
        cordinate_splits = cordinate_text.split(',')
        for cordinate_split in cordinate_splits:
            cordinate.append(int(cordinate_split))

        img = cv2.imread('./gt/' + file[:-5]+'.jpg')
        # result_data = self.result_datas[current_file]
        img=self.gen_result_img(img,file,1)
        cv2.rectangle(img, (cordinate[0],cordinate[1]), (cordinate[2],cordinate[3]), (0, 255, 255), 5)
        cv2.putText(img, gt_label, (cordinate[0],cordinate[1]), cv2.FONT_HERSHEY_PLAIN , 2, (0, 255, 255),2)

        # cv2.imwrite("./img/"+ file[:-5]+'.jpg' ,img)
        qt_img=self.convert_cv_qt(img,640,360)
        self.label_screen_false.setPixmap(qt_img)



    def tableWidget_filelist_doubleClicked(self):
        row = self.tableWidget_filelist.currentIndex().row()
        column = self.tableWidget_filelist.currentIndex().column()
      
        gt_label=self.tableWidget_filelist.item(row, 2).text()
        
        cordinate=[]
        cordinate_text=self.tableWidget_filelist.item(row, 3).text()
        cordinate_text= cordinate_text.strip("]""[")
        cordinate_splits = cordinate_text.split(',')
        for cordinate_split in cordinate_splits:
            cordinate.append(int(cordinate_split))

        draw_img=self.current_img.copy()

        cv2.rectangle(draw_img, (cordinate[0],cordinate[1]), (cordinate[2],cordinate[3]), (0, 255, 255), 5)
        cv2.putText(draw_img, gt_label, (cordinate[0],cordinate[1]), cv2.FONT_HERSHEY_PLAIN , 2, (0, 255, 255),2)
        # cv2.imwrite("./img/"+ file[:-5]+'.jpg' ,img)
        qt_img=self.convert_cv_qt(draw_img,1280,720)
        self.label_screen.setPixmap(qt_img)





    def gen_result_img(self, img , current_file,thick,filelist_state=False):

        tp_boxs = self.result_datas[current_file]["resultbox"]["tp_box"]
        e2e_tp_boxs = self.result_datas[current_file]["resultbox"]["e2e_tp_box"]
        fn_boxs = self.result_datas[current_file]["resultbox"]["fn_box"]
        gt_boxs = self.result_datas[current_file]["resultbox"]["gt_box"]
        e2e_gt_boxs = self.result_datas[current_file]["resultbox"]["e2e_gt_box"]
        over_gt_boxs = self.result_datas[current_file]["resultbox"]["over_gt_box"]
        overlab_boxs = self.result_datas[current_file]["resultbox"]["overlab_box"]
        e2e_overlab_boxs = self.result_datas[current_file]["resultbox"]["e2e_overlab_box"]
        fp_boxs = self.result_datas[current_file]["resultbox"]["fp_box"]

        if filelist_state:
            self.tableWidget_filelist.setRowCount(0)
            self.textEdit_file_result.clear(  )

            det_precision = str(self.result_datas[current_file]['eval_score']['det_precision'])
            det_recall = str(self.result_datas[current_file]['eval_score']['det_recall'])
            e2e_precision = str(self.result_datas[current_file]['eval_score']['e2e_precision'])
            e2e_recall = str(self.result_datas[current_file]['eval_score']['e2e_recall'])

            self.textEdit_file_result.append("img: " + current_file[:-5])
            self.textEdit_file_result.append("det_precision: " + det_precision)
            self.textEdit_file_result.append("det_recall: " + det_recall)
            self.textEdit_file_result.append("e2e_precision: " + e2e_precision)
            self.textEdit_file_result.append("e2e_recall: " + e2e_recall)

         

        for bbox in fn_boxs:
            img = self.pain_bbox(img, bbox, (20, 0, 120), -1) # fn_list
            if filelist_state:
                self.tableWidget_filelist.insertRow(self.tableWidget_filelist.rowCount())
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 0, QTableWidgetItem('FN'))
                nan_item=QTableWidgetItem('NaN')
                nan_item.setForeground(QBrush(QColor(128, 128, 128)))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 1, nan_item)
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 2, QTableWidgetItem(str(bbox[0])))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 3, QTableWidgetItem(str(bbox[1:5])))
            
        for bbox in gt_boxs:
            img = self.pain_bbox(img, bbox, (0, 60, 20), -1) # false gt
            if filelist_state:
                self.tableWidget_filelist.insertRow(self.tableWidget_filelist.rowCount())
                
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 0, QTableWidgetItem('TP'))
                false_item=QTableWidgetItem('False')
                false_item.setForeground(QBrush(QColor(255, 0, 0)))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 1, false_item)
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 2, QTableWidgetItem(str(bbox[0])))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 3, QTableWidgetItem(str(bbox[1:5])))

        for bbox in e2e_gt_boxs:
            img = self.pain_bbox(img, bbox, (0, 60, 20), -1) # true gt

        for bbox in tp_boxs:
            cv2.rectangle(img, (bbox[1],bbox[2]), (bbox[3],bbox[4]), (0, 0, 255), thick)
            cv2.putText(img, bbox[0], (bbox[1],bbox[2]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),1)

        for bbox in e2e_tp_boxs:
            cv2.rectangle(img, (bbox[1],bbox[2]), (bbox[3],bbox[4]), (0, 255, 0), thick)
            cv2.putText(img, bbox[0], (bbox[1],bbox[2]), cv2.FONT_HERSHEY_PLAIN , 1, (0, 255, 0),1)

        for bbox in e2e_overlab_boxs:
            cv2.rectangle(img, (bbox[1],bbox[2]), (bbox[3],bbox[4]), (255, 0, 0), thick)
            cv2.putText(img, bbox[0],(bbox[1],bbox[2]), cv2.FONT_HERSHEY_PLAIN , 1, (255, 0, 0),1)
  
        for bbox in overlab_boxs:
            cv2.rectangle(img, (bbox[1],bbox[2]), (bbox[3],bbox[4]), (255, 0, 0), thick)
            cv2.putText(img, bbox[0], (bbox[1],bbox[2]), cv2.FONT_HERSHEY_PLAIN , 1, (255, 0, 0),1)
        
        for bbox in over_gt_boxs:
            img = self.pain_bbox(img, bbox, (100, 20, 0), -1) # fn_list
            if filelist_state:
                self.tableWidget_filelist.insertRow(self.tableWidget_filelist.rowCount())
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 0, QTableWidgetItem('Over'))
                nan_item=QTableWidgetItem('NaN')
                nan_item.setForeground(QBrush(QColor(128, 128, 128)))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 1, nan_item)
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 2, QTableWidgetItem(str(bbox[0])))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 3, QTableWidgetItem(str(bbox[1:5])))

        for bbox in fp_boxs:
            cv2.rectangle(img, (bbox[1],bbox[2]), (bbox[3],bbox[4]), (0, 0, 255), thick)
            cv2.putText(img, bbox[0], (bbox[1],bbox[2]), cv2.FONT_HERSHEY_PLAIN , 1, (0, 0, 255),1)
            if filelist_state:
                self.tableWidget_filelist.insertRow(self.tableWidget_filelist.rowCount())
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 0, QTableWidgetItem('FP'))
                nan_item=QTableWidgetItem('NaN')
                nan_item.setForeground(QBrush(QColor(128, 128, 128)))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 1, nan_item)
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 2, QTableWidgetItem(str(bbox[0])))
                self.tableWidget_filelist.setItem(self.tableWidget_filelist.rowCount()-1, 3, QTableWidgetItem(str(bbox[1:5])))


        table = self.tableWidget_filelist
        header = table.horizontalHeader()
        twidth = header.width()
        width = []
        for column in range(header.count()):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
            width.append(header.sectionSize(column))

        wfactor = twidth / sum(width)
        for column in range(header.count()):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
            header.resizeSection(column, width[column]*wfactor)



        return img






    def listWidget_filelist_DoubleClicked(self):
        self.current_file=self.listWidget_filelist.currentItem().text()
        current_row = self.listWidget_filelist.currentRow()
        self.current_img = cv2.imread('./gt/' + self.current_file[:-5]+'.jpg')

        # result_data = self.result_datas[current_file]
        self.current_img=self.gen_result_img(self.current_img,self.current_file,2,True)

        # cv2.imwrite("./img/"+ file[:-5]+'.jpg'    ,img)
        qt_img=self.convert_cv_qt(self.current_img,1280,720)
        self.label_screen.setPixmap(qt_img)



    def print_result(self, data, class_label):
        
        # self.calc = External()
        # self.calc.countChanged.connect(self.onCountChanged)
        # self.calc.start()


        self.tableWidget_gt.setRowCount(0)
        self.textEdit_summary.clear(  )
        self.listWidget_filelist.clear()

        pj_name=data["project_name"] 
        iou=data["IOU"] 
        self.result_datas=data["result"]
        summary_data=data["summary"]

        tp_boxs = []
        e2e_tp_boxs = []
        fn_boxs = []
        gt_boxs = []
        e2e_gt_boxs = []
        over_gt_boxs = []
        overlab_boxs = []
        e2e_overlab_boxs= []
        fp_boxs= []

        df = pd.DataFrame(columns = ['label','x1','y1','x2','y2','filename','predictlabel','iou','box','gt_predict','true'])
        df2 = pd.DataFrame(columns = ['label','x1','y1','x2','y2','filename','predictlabel','iou','box','gt_predict','true'])
        gt_det_label_df = pd.DataFrame(columns = ['gt_label' , 'det_label' , 'iou'])
       
        for result_file_data in self.result_datas:
            file = result_file_data
            result_data = self.result_datas[result_file_data]
            self.listWidget_filelist.addItem(file)
            
            tp_boxs = tp_boxs + result_data["resultbox"]["tp_box"]
            e2e_tp_boxs = e2e_tp_boxs+ result_data["resultbox"]["e2e_tp_box"]
            fn_boxs = fn_boxs+ result_data["resultbox"]["fn_box"]
            gt_boxs =  gt_boxs+ result_data["resultbox"]["gt_box"]
            over_gt_boxs = over_gt_boxs+ result_data["resultbox"]["over_gt_box"]
            e2e_gt_boxs =  e2e_gt_boxs+ result_data["resultbox"]["e2e_gt_box"]
            overlab_boxs = overlab_boxs+ result_data["resultbox"]["overlab_box"]
            e2e_overlab_boxs = e2e_overlab_boxs+ result_data["resultbox"]["e2e_overlab_box"]
            fp_boxs = fp_boxs+ result_data["resultbox"]["fp_box"]

     





        df = pd.DataFrame()

        if len(tp_boxs) > 0:
            tempdf = pd.DataFrame(tp_boxs)
            tempdf.insert(6,'predictlabel','NaN')
            tempdf.insert(7,'iou','NaN')
            tempdf.insert(8,'box','TP')
            tempdf.insert(9,'gt_predict','predict')
            tempdf.insert(10,'true','False')
            tempdf.columns=['label','x1','y1','x2','y2','filename','predictlabel','iou','box','gt_predict','true']
            df = pd.concat([df2,tempdf],ignore_index=True)


        # for e2e_tp_box in tqdm(e2e_tp_boxs):
        #     df=df.append({'label' : e2e_tp_box[0] , 'box' : 'TP', 'point' : e2e_tp_box[1:5], 'gt_predict': 'predict', 'true': 'True'} , ignore_index=True)
        
   
        if len(e2e_tp_boxs) > 0:
            tempdf = pd.DataFrame(e2e_tp_boxs)
            tempdf.insert(6,'predictlabel','NaN')
            tempdf.insert(7,'iou','NaN')
            tempdf.insert(8,'box','TP')
            tempdf.insert(9,'gt_predict','predict')
            tempdf.insert(10,'true','True')
            tempdf.columns=['label','x1','y1','x2','y2','filename','predictlabel','iou','box','gt_predict','true']
            df = pd.concat([df,tempdf],ignore_index=True)

    
        # for fn_box in tqdm(fn_boxs):
        #     df=df.append({'label' : fn_box[0] , 'box' : 'FN', 'point' : fn_box[1:5], 'gt_predict': 'gt'} , ignore_index=True)
        #     # self.tableWidget_det.insertRow(self.tableWidget_det.rowCount())
        #     # self.tableWidget_det.setItem(self.tableWidget_det.rowCount()-1, 0, QTableWidgetItem('FN'))
        #     # self.tableWidget_det.setItem(self.tableWidget_det.rowCount()-1, 1, QTableWidgetItem(str(fn_box[0])))
        #     # self.tableWidget_det.setItem(self.tableWidget_det.rowCount()-1, 2, QTableWidgetItem(str(fn_box[1:5])))
        #     # self.tableWidget_det.setItem(self.tableWidget_det.rowCount()-1, 3, QTableWidgetItem(str(fn_box[5])))
        

        if len(fn_boxs) > 0:
            tempdf = pd.DataFrame(fn_boxs)
            tempdf.insert(6,'predictlabel','NaN')
            tempdf.insert(7,'iou','NaN')
            tempdf.insert(8,'box','FN')
            tempdf.insert(9,'gt_predict','gt')
            tempdf.insert(10,'true','NaN')
            tempdf.columns=['label','x1','y1','x2','y2','filename','predictlabel','iou','box','gt_predict','true']
            df = pd.concat([df,tempdf],ignore_index=True)



        for gt_box in gt_boxs:
            # df=df.append({'label' : gt_box[0] , 'box' : 'GT', 'point' : gt_box[1:5], 'gt_predict': 'gt', 'true': 'False'} , ignore_index=True)
            # gt_det_label_df=gt_det_label_df.append({'gt_label':gt_box[0], 'det_label':gt_box[6], 'iou': gt_box[7]}, ignore_index=True)
            self.tableWidget_gt.insertRow(self.tableWidget_gt.rowCount())
            false_item=QTableWidgetItem('False')
            false_item.setForeground(QBrush(QColor(255, 0, 0)))
            self.tableWidget_gt.setItem(self.tableWidget_gt.rowCount()-1, 0, false_item)
            self.tableWidget_gt.setItem(self.tableWidget_gt.rowCount()-1, 1, QTableWidgetItem(str(gt_box[0])))
            self.tableWidget_gt.setItem(self.tableWidget_gt.rowCount()-1, 2, QTableWidgetItem(str(gt_box[6])))
            self.tableWidget_gt.setItem(self.tableWidget_gt.rowCount()-1, 3, QTableWidgetItem(str(gt_box[5])))
            self.tableWidget_gt.setItem(self.tableWidget_gt.rowCount()-1, 4, QTableWidgetItem(str(gt_box[1:5])))
        

        if len(gt_boxs) > 0:
            tempdf = pd.DataFrame(gt_boxs)
            tempdf.insert(8,'box','GT')
            tempdf.insert(9,'gt_predict','gt')
            tempdf.insert(10,'true','False')
            tempdf.columns=['label','x1','y1','x2','y2','filename','predictlabel','iou','box','gt_predict','true']

            df = pd.concat([df,tempdf],ignore_index=True)

            tempdf.drop(['x1','y1','x2','y2','filename','box','gt_predict','true'], axis=1,inplace=True)
            tempdf.columns = ['gt_label' , 'det_label' , 'iou']
            gt_det_label_df = pd.concat([gt_det_label_df,tempdf],ignore_index=True)

        # for over_gt_box in tqdm(over_gt_boxs):
        #     df=df.append({'label' : over_gt_box[0] , 'box' : 'OVER', 'point' : over_gt_box[1:5], 'gt_predict': 'gt'} , ignore_index=True)
        

        if len(over_gt_boxs) > 0:
            tempdf = pd.DataFrame(over_gt_boxs)
            tempdf.insert(6,'predictlabel','NaN')
            tempdf.insert(7,'iou','NaN')
            tempdf.insert(8,'box','OVER')
            tempdf.insert(9,'gt_predict','gt')
            tempdf.insert(10,'true','NaN')
            tempdf.columns=['label','x1','y1','x2','y2','filename','predictlabel','iou','box','gt_predict','true']
            df = pd.concat([df,tempdf],ignore_index=True)



        # for e2e_gt_box in tqdm(e2e_gt_boxs):
        #     df=df.append({'label' : e2e_gt_box[0] , 'box' : 'GT', 'point' : e2e_gt_box[1:5], 'gt_predict': 'gt', 'true': 'True'} , ignore_index=True)
        #     gt_det_label_df=gt_det_label_df.append({'gt_label':e2e_gt_box[0], 'det_label':e2e_gt_box[6], 'iou': e2e_gt_box[7]}, ignore_index=True)
        

        if len(e2e_gt_boxs) > 0:
            tempdf = pd.DataFrame(e2e_gt_boxs)
            tempdf.insert(8,'box','GT')
            tempdf.insert(9,'gt_predict','gt')
            tempdf.insert(10,'true','True')
            tempdf.columns=['label','x1','y1','x2','y2','filename','predictlabel','iou','box','gt_predict','true']
            df = pd.concat([df,tempdf],ignore_index=True)


            tempdf.drop(['x1','y1','x2','y2','filename','box','gt_predict','true'], axis=1,inplace=True)
            tempdf.columns = ['gt_label' , 'det_label' , 'iou']
            gt_det_label_df = pd.concat([gt_det_label_df,tempdf],ignore_index=True)

        # for overlab_box in tqdm(overlab_boxs):
        #     df=df.append({'label' : overlab_box[0] , 'box' : 'OVER', 'point' : overlab_box[1:5], 'gt_predict': 'predict', 'true': 'False'} , ignore_index=True)
        

  
        if len(overlab_boxs) > 0:
            tempdf = pd.DataFrame(overlab_boxs)
 
            tempdf.insert(6,'predictlabel','NaN')
            tempdf.insert(7,'iou','NaN')
            tempdf.insert(8,'box','OVER')
            tempdf.insert(9,'gt_predict','predict')
            tempdf.insert(10,'true','False')
            tempdf.columns=['label','x1','y1','x2','y2','filename','predictlabel','iou','box','gt_predict','true']

            df = pd.concat([df,tempdf],ignore_index=True)

    

        # for e2e_overlab_box in tqdm(e2e_overlab_boxs):
        #     df=df.append({'label' : e2e_overlab_box[0] , 'box': 'OVER', 'point' : e2e_overlab_box[1:5], 'gt_predict': 'predict', 'true': 'True'} , ignore_index=True)
        
        if len(e2e_overlab_boxs) > 0:
            tempdf = pd.DataFrame(e2e_overlab_boxs)
            tempdf.insert(6,'predictlabel','NaN')
            tempdf.insert(7,'iou','NaN')
            tempdf.insert(8,'box','OVER')
            tempdf.insert(9,'gt_predict','predict')
            tempdf.insert(10,'true','true')
            tempdf.columns=['label','x1','y1','x2','y2','filename','predictlabel','iou','box','gt_predict','true']

            df = pd.concat([df,tempdf],ignore_index=True)


        # for fp_box in tqdm(fp_boxs):
        #     df=df.append({'label' : fp_box[0] , 'box' : 'FP', 'point' : fp_box[1:5], 'gt_predict': 'predict'} , ignore_index=True)
        #     # self.tableWidget_det.insertRow(self.tableWidget_det.rowCount())
        #     # self.tableWidget_det.setItem(self.tableWidget_det.rowCount()-1, 0, QTableWidgetItem('FP'))
        #     # self.tableWidget_det.setItem(self.tableWidget_det.rowCount()-1, 1, QTableWidgetItem(str(fp_box[0])))
        #     # self.tableWidget_det.setItem(self.tableWidget_det.rowCount()-1, 2, QTableWidgetItem(str(fp_box[1:5])))
        #     # self.tableWidget_det.setItem(self.tableWidget_det.rowCount()-1, 3, QTableWidgetItem(str(fp_box[5])))

        if len(fp_boxs) > 0:
            tempdf = pd.DataFrame(fp_boxs)
            tempdf.insert(6,'predictlabel','NaN')
            tempdf.insert(7,'iou','NaN')
            tempdf.insert(8,'box','FP')
            tempdf.insert(9,'gt_predict','predict')
            tempdf.insert(10,'true','NaN')
            tempdf.columns=['label','x1','y1','x2','y2','filename','predictlabel','iou','box','gt_predict','true']

            df = pd.concat([df,tempdf],ignore_index=True)




        table = self.tableWidget_gt
        header = table.horizontalHeader()
        twidth = header.width()
        width = []
        for column in range(header.count()):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
            width.append(header.sectionSize(column))

        wfactor = twidth / sum(width)
        for column in range(header.count()):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
            header.resizeSection(column, width[column]*wfactor)


        gt_label_df = df.loc[df.gt_predict == 'gt']
        # gt_label_df = df.loc[df.gt_predict == 'gt'].loc[df.box != 'GT']
        gt_label_df_False = gt_label_df.loc[gt_label_df.true == 'False']


        det_label_df = df.loc[df.gt_predict == 'predict']
        # det_label_df = df.loc[df.gt_predict == 'predict'].loc[df.box != 'TP']
        det_label_df_False =  det_label_df.loc[det_label_df.box != 'OVER']
    
        
        if len(df) > 0:
            self.fig = plt.figure()
            self.canvas = FigureCanvas(self.fig)
            df = df.sort_values(by='label' ,ascending=True)
            plot3 = sns.countplot(data=df, y='label', hue ='gt_predict', palette='Set2')
            plot3.set_title("GT / Prediction")
            self.clearlayout(self.gridLayout_3)
            self.gridLayout_3.addWidget(self.canvas)
            self.canvas.draw()
        

     

        df2=pd.concat([det_label_df_False,gt_label_df])
        df3=df2.loc[df2.box != 'TP'].loc[df2.box != 'GT']
        



        if len(df3) > 0:
            self.fig2 = plt.figure()
            self.canvas2 = FigureCanvas(self.fig2)
            df3 = df3.sort_values(by='label' ,ascending=True)
            plot2 = sns.countplot(data=df3, y='label', hue ='box', palette='Set3')
            plot2.set_title("Error Type : Detection")

            for p in plot2.patches:
                height = p.get_height()
                width= p.get_width()
                if width > 0:
                    plot2.text(p.get_x()+width+3, p.get_y()+(height/2), int(width), ha = 'center',va = 'center', size = 8, color = 'black')

            self.clearlayout(self.gridLayout_4)
            self.gridLayout_4.addWidget(self.canvas2)
            self.canvas2.draw()


        
        if len(gt_det_label_df) > 0:
            self.fig3 = plt.figure()
            self.canvas3 = FigureCanvas(self.fig3)
            gt_det_label_df = gt_det_label_df.sort_values(by='gt_label' ,ascending=True)
            # sns.stripplot(data=gt_det_label_df, y='gt_label', hue ='iou', palette='Set2')
            plot5 = sns.stripplot(data=gt_det_label_df, x='iou', y='gt_label')
            plot5.set_title("IOU : Detection")
            self.clearlayout(self.gridLayout_5)
            self.gridLayout_5.addWidget(self.canvas3)
            self.canvas3.draw()



        # if len(gt_label_df_False):
        if len(gt_label_df_False) > 0:
            self.fig4 = plt.figure()
            self.canvas4 = FigureCanvas(self.fig4)
            gt_label_df_False = gt_label_df_False.sort_values(by='label' ,ascending=True)
            plot6 = sns.countplot(data=gt_label_df_False, y='label', palette='Set3')
            plot6.set_title("Recognition False")

            for p in plot6.patches:
                height = p.get_height()
                width= p.get_width()
                plot6.text(p.get_x()+width, p.get_y()+(height/2), width, ha = 'center',va = 'center', size = 8, color = 'black')

            self.clearlayout(self.gridLayout_6)
            self.gridLayout_6.addWidget(self.canvas4)
            self.canvas4.draw()

        class_label.sort()
        tp_heatmap_df = pd.DataFrame(0, index=class_label, columns=class_label)
        for idx,row in gt_det_label_df.iterrows():
            tp_heatmap_df.at[row['gt_label'],row['det_label']] += 1

   
        self.fig5 = plt.figure()
        self.canvas5 = FigureCanvas(self.fig5)
        # gt_det_label_df = gt_det_label_df.sort_values(by='gt' ,ascending=True)
        plot7= sns.heatmap(data=tp_heatmap_df,square=True, cmap='Blues', annot = True,fmt="d", cbar=False,linewidths=0.5,annot_kws={"size": 8})
        plot7.set_title("Recognition Heatmap")

        self.clearlayout(self.gridLayout_7)
        self.gridLayout_7.addWidget(self.canvas5)
        self.canvas5.draw()

   
        evalname = self.lineEdit_evalname.text()
        self.textEdit_summary.append("project: " + str(pj_name))
        self.textEdit_summary.append("IOU: " + str(iou))
        self.textEdit_summary.append("detection_precision: " + str(summary_data["total_det_precision"]))
        self.textEdit_summary.append("detection_recall: " + str(summary_data["total_det_recall"]))
        self.textEdit_summary.append("detection_Hmean: " + str(summary_data["det_Hmean"]))
        self.textEdit_summary.append("e2e_precision: " + str(summary_data["total_e2e_precision"]))
        self.textEdit_summary.append("e2e_recall: " + str(summary_data["total_e2e_recall"]))
        self.textEdit_summary.append("e2e_Hmean: " + str(summary_data["e2e_Hmean"]))

        self.label_pj_name.setText(evalname)
        self.label_det_pre.setText(str(summary_data["total_det_precision"]))
        self.label_det_recall.setText(str(summary_data["total_det_recall"]))

        self.label_pj_name2.setText(evalname)
        self.label_e2e_pre.setText(str(summary_data["total_e2e_precision"]))
        self.label_e2e_recall.setText(str(summary_data["total_e2e_recall"]))


    def clearlayout(self,layout):
        for i in reversed(range(layout.count())):
            layout.removeItem(layout.itemAt(i))


    @pyqtSlot(dict,list)
    def outevalresult(self, result_data, class_label):
        self.print_result(result_data, class_label)


    @pyqtSlot(int,int)
    def onCountChanged(self, value,value2):
        p = value+1
        e = value2
        self.progressBar.setValue(int(p/e*100))


    def pushButton_evaluation_fuction(self):
        
        self.calc = External(self)
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.outevalresult.connect(self.outevalresult)
        self.calc.start()

        self.pushButton_evaluation.setEnabled(False)
        
    
        # iou_th = self.doubleSpinBox_iou.value()
        
        # iou_th=round(iou_th,3)
        # try:
        #     gt_path = self.gt_path
        #     det_path = self.det_path
        # except:
        #     QMessageBox.information(self,'오류','경로를 지정하시오')
        #     return


        # total_gt_list = []
        # total_over_gt_list = []
        # total_e2e_gt_list = []
        # total_fn_list = []

        # total_tp_list = []
        # total_e2e_tp_list = []
        # total_fp_list = []
        # total_overlab_list = []
        # total_e2e_overlab_list = []

        # class_label= []

        # data = {}

        # fileio = Fileio()
        # eval = Eval()

        # gtbox_list = []
        # detbox_list = []

        # # try:
        # gtfile_list =os.listdir(self.gt_path)
        # gtfile_list = [file for file in gtfile_list if file.endswith(".json")]
        # for gtfile in tqdm(gtfile_list):
        #     detfile = gtfile
        #     gtbox_list = fileio.read_file(self.gt_path+'/'+ gtfile, gtfile)
        #     detbox_list = fileio.read_file(self.det_path+'/'+ detfile, gtfile)

        #     resultbox , score, label= eval.evaluation(gtbox_list, detbox_list, iou_th)
        #     data[gtfile] = {
        #         "resultbox" : resultbox, 
        #         "eval_score" : score
        #     }

        #     class_label += label
        #     total_tp_list += resultbox['tp_box']
        #     total_e2e_tp_list += resultbox['e2e_tp_box']
        #     total_fp_list +=resultbox['fp_box']
        #     total_fn_list += resultbox['fn_box']
        #     total_gt_list += resultbox['gt_box']
        #     total_over_gt_list += resultbox['over_gt_box']
        #     total_e2e_gt_list += resultbox['e2e_gt_box']
        #     total_overlab_list += resultbox['overlab_box']
        #     total_e2e_overlab_list += resultbox['e2e_overlab_box']

        # # except:
        # #     QMessageBox.information(self,'오류','파일 오류')
        # #     return


        # class_label_set = set(class_label) #집합set으로 변환
        # class_label = list(class_label_set)

        # total_det_n = len(total_tp_list)+ len(total_e2e_tp_list) + len(total_overlab_list) + len(total_e2e_overlab_list) + len(total_fp_list)
        # total_gt_n = len(total_fn_list) + len(total_gt_list) + len(total_e2e_gt_list) + len(total_over_gt_list)

        # total_det_precision =  round((len(total_tp_list + total_e2e_tp_list) / total_det_n),4)
        # total_det_recall = round((len(total_tp_list + total_e2e_tp_list) / total_gt_n),4)
        # total_e2e_precision =  round((len(total_e2e_tp_list) / total_det_n),4)
        # total_e2e_recall = round((len(total_e2e_tp_list) / total_gt_n),4)

        
        # det_Hmean = round(2 * (total_det_precision * total_det_recall) / (total_det_precision + total_det_recall),4)
        # e2e_Hmean = round(2 * (total_e2e_precision * total_e2e_recall) / (total_e2e_precision + total_e2e_recall),4)


        # result_data = {}
        # now = datetime.now()
        # result_data["project_name"] = self.lineEdit_evalname.text()
        # result_data["date"] = str(now)
        # result_data["class_label"] = class_label
        # result_data["IOU"] = iou_th
        # result_data["result"] = data
        # result_data["summary"] = {
        #     "total_det_precision" : total_det_precision,
        #     "total_det_recall": total_det_recall,
        #     "det_Hmean" : det_Hmean,
        #     "total_e2e_precision" : total_e2e_precision,
        #     "total_e2e_recall": total_e2e_recall,
        #     "e2e_Hmean" : e2e_Hmean
        # }

        # filename = self.lineEdit_evalname.text() + '.json'
        # json_dump(result_data,filename)
        # self.print_result(result_data, class_label)

    def convert_cv_qt(self, cv_img, disply_width, display_height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(disply_width, display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def pain_bbox(self, img, bbox, color, thickness):
        bbox_img = np.full((img.shape[0], img.shape[1], 3), (0, 0, 0), dtype=np.uint8)
        cv2.rectangle(bbox_img, (bbox[1],bbox[2]), (bbox[3],bbox[4]), color, thickness)
        if thickness == -1:
            bboxs_img = np.full((img.shape[0], img.shape[1], 3), (0, 0, 0), dtype=np.uint8)
            cv2.rectangle(bboxs_img,(bbox[1],bbox[2]), (bbox[3],bbox[4]), ((100-color[0])/2 ,(100-color[1])/2,(100-color[2])/2), thickness)
            img = cv2.subtract(img, bboxs_img)
        img = cv2.add(img, bbox_img)
        return img


if __name__ == "__main__":

    app = QApplication(sys.argv) 

    myWindow = WindowClass() 

    myWindow.show()

    app.exec_()