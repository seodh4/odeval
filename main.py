import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import qdarkstyle
from eval import *
import numpy as np
import cv2

from PyQt5.QtCore import pyqtSlot, Qt 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtGui




form_class = uic.loadUiType("windows.ui")[0]

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        iou_th = 0.2

        fileio = Fileio()
        eval = Eval()

        self.gtbox_list = []
        self.detbox_list = []

        # gtfile_list =os.listdir('./gt/')
        # gtfile_list = [file for file in gtfile_list if file.endswith(".txt")]
        # for gtfile in gtfile_list:
        #     detfile = gtfile
        gtfile = 'cam0_20220119_083120_4.json'
        detfile = 'cam0_20220119_083120_4.json'
        self.gtbox_list = fileio.read_file('./gt/'+ gtfile)
        self.detbox_list = fileio.read_file('./det/'+ detfile)



        gtbox_n = len(self.gtbox_list)
        detbox_n = len(self.detbox_list)

        evalbox = np.zeros(shape=(gtbox_n,detbox_n))


        self.tableWidget.setRowCount(gtbox_n)
        self.tableWidget.setColumnCount(detbox_n)
        self.tableWidget.cellClicked.connect(self.cell_clicked)
        # self.tableWidget.verticalHeader().sectionClicked.connect(self.verticalHeader_clicked)
        # self.tableWidget.HorizontalHeader().sectionClicked.connect(self.HorizontalHeader_clicked)


        self.img = cv2.imread('./gt/' + gtfile[:-5]+'.jpg')
        



        for gtidx, gtbox in enumerate(self.gtbox_list):
            gtbox_coordinate = gtbox[1:]
            gtbox_label = gtbox[0]

            for detidx, detbox in enumerate(self.detbox_list):
                detbox_coordinate = detbox[1:]
                detbox_label = detbox[0]

                # eval.IoU(gtbox_coordinate,detbox_coordinate)
                iou = eval.IoU(gtbox_coordinate,detbox_coordinate)
                evalbox[gtidx][detidx] = iou

                item = QTableWidgetItem(str(iou))

                # item.setForeground(QBrush(QColor(0, 255, 0)))
                # if iou > 0.5:
                #     item.setForeground(QBrush(Qt.green))
                #     # item.setBackground(Qt.red)

                self.tableWidget.setItem(gtidx, detidx, item)



        gt_list = []
        det_list=[]

        tp_list = []
        e2e_tp_list = []
        fp_list = []
        fn_list = []

        overlab_list = []
        e2e_overlab_list = []




        for gt_idx, det_eval_list in enumerate(evalbox):
            

            label = self.gtbox_list[gt_idx][0]

            det_TP_idx_list = np.where(det_eval_list > iou_th)[0]

            #FN
            if len(det_TP_idx_list) == 0:
                fn_list.append(self.gtbox_list[gt_idx])
                self.img = self.pain_bbox(self.img, self.gtbox_list[gt_idx], (20, 0, 120), -1)
            else:
                gt_list.append(self.gtbox_list[gt_idx])
                self.img = self.pain_bbox(self.img, self.gtbox_list[gt_idx], (0, 60, 20), -1)

            #overlab
            if len(det_TP_idx_list) > 1:
                for det_TP_idx in det_TP_idx_list:
                    # self.tableWidget.item(gt_idx, det_TP_idx).setForeground(QBrush(QColor(0, 0, 100)))
                    overlab_list.append(self.detbox_list[det_TP_idx])
                    
                    if label == self.detbox_list[det_TP_idx][0]: 
                        e2e_overlab_list.append(self.detbox_list[det_TP_idx])
                        self.tableWidget.item(gt_idx, det_TP_idx).setBackground(QBrush(QColor(180, 255, 180)))
                        cv2.rectangle(self.img, self.detbox_list[det_TP_idx][1:3], self.detbox_list[det_TP_idx][3:5], (0, 255, 0), 2)
                        cv2.putText(self.img, self.detbox_list[det_TP_idx][0], self.detbox_list[det_TP_idx][1:3], cv2.FONT_HERSHEY_PLAIN , 1, (0, 255, 0),2)
                    else:
                        self.tableWidget.item(gt_idx, det_TP_idx).setBackground(QBrush(QColor(255, 180, 180)))
                        cv2.rectangle(self.img, self.detbox_list[det_TP_idx][1:3], self.detbox_list[det_TP_idx][3:5], (0, 0, 255), 2)
                        cv2.putText(self.img, self.detbox_list[det_TP_idx][0], self.detbox_list[det_TP_idx][1:3], cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),2)

            #TP
            if len(det_TP_idx_list) == 1:

    
                det_TP_idx = det_TP_idx_list[0]

                self.tableWidget.item(gt_idx, det_TP_idx).setForeground(QBrush(QColor(0, 100, 0)))
                tp_list.append(self.detbox_list[det_TP_idx])

                if label == self.detbox_list[det_TP_idx][0]:
                    e2e_tp_list.append(self.detbox_list[det_TP_idx])
                    self.tableWidget.item(gt_idx, det_TP_idx).setBackground(QBrush(QColor(180, 255, 180)))
                    cv2.rectangle(self.img, self.detbox_list[det_TP_idx][1:3], self.detbox_list[det_TP_idx][3:5], (0, 255, 0), 2)
                    cv2.putText(self.img, self.detbox_list[det_TP_idx][0], self.detbox_list[det_TP_idx][1:3], cv2.FONT_HERSHEY_PLAIN , 1, (0, 255, 0),2)
                else:
                    self.tableWidget.item(gt_idx, det_TP_idx).setBackground(QBrush(QColor(255, 180, 180)))
                    cv2.rectangle(self.img, self.detbox_list[det_TP_idx][1:3], self.detbox_list[det_TP_idx][3:5], (0, 0, 255), 2)
                    cv2.putText(self.img, self.detbox_list[det_TP_idx][0], self.detbox_list[det_TP_idx][1:3], cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),2)
                    


   
        for det_idx, gt_eval_list in enumerate(evalbox.T):

            label = self.detbox_list[det_idx][0]

            gt_TP_idx_list = np.where(gt_eval_list > iou_th)[0]

            #FP
            if len(gt_TP_idx_list) == 0:
                fp_list.append(self.detbox_list[det_idx])
                cv2.rectangle(self.img, self.detbox_list[det_idx][1:3], self.detbox_list[det_idx][3:5], (0, 0, 255), 2)
                cv2.putText(self.img, self.detbox_list[det_idx][0], self.detbox_list[det_idx][1:3], cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),2)
               
        

        print(fp_list)
        print(tp_list)
        print(e2e_tp_list)
        print(fn_list)
        print(overlab_list)


        print(len(tp_list)+ len(overlab_list) + len(fp_list)) # det

        print(len(fn_list) +len(tp_list) + (len(overlab_list)/2)) #gt


        # print(e2e_overlab_list)


        qt_img=self.convert_cv_qt(self.img,1280,720)
        self.label_screen.setPixmap(qt_img)

            # print(self.gtbox_list)

            # len_self.gtbox_list=len([x for x in self.gtbox_list if x != ''])
            # print(len_self.gtbox_list)


            # det_eval_list = [x for x in gt if x > iou_th]
            # maxiou = max(det_eval_list)
            # maxiou_idx = np.where(gt == maxiou)[0][0]



            # print(maxiou_idx)
            # print(self.detbox_list[maxiou_idx])


            # tp_idx_list = np.where(gt > 0.5)
            # print(tp_idx_list)
            # print(self.gtbox_list[tp_idx_list[0]])


            # det_tp_list = [x for x in gt if x > 0.5]
            # print(det_tp_list)


            
            # a = np.where(gt > 0.5)
            # print(a)

                # if iou > 0.5:
                #     pass
                #     print(gtbox_label, detbox_label)



        # print(det_evalbox)
                    
    # def verticalHeader_clicked(self, row):
    #     canvas_img =self.img.copy()
    #     canvas_img = self.pain_bbox(canvas_img, self.gtbox_list[row], (0, 100, 100), -1)
    #     qt_img=self.convert_cv_qt(canvas_img,1280,720)
    #     self.label_screen.setPixmap(qt_img)

    # def HorizontalHeader_clicked(self, column):
    #     canvas_img =self.img.copy()
    #     cv2.rectangle(canvas_img, self.detbox_list[column][1:3], self.detbox_list[column][3:5], (0, 255, 255), 2)
    #     qt_img=self.convert_cv_qt(canvas_img,1280,720)
    #     self.label_screen.setPixmap(qt_img)

    def cell_clicked(self, row, column):
        # item = self.tableWidget.item(row, column)
        # value = item.text()
        # label_string = 'Row: ' + str(row+1) + ', Column: ' + str(column+1) + ', Value: ' + str(value)
        # print(label_string)
        # self.label.setText(label_string)
        canvas_img =self.img.copy()
        # cv2.rectangle(canvas_img, self.gtbox_list[row][1:3], self.gtbox_list[row][3:5], (0, 255, 255), 3)
        canvas_img = self.pain_bbox(canvas_img, self.gtbox_list[row], (0, 100, 100), -1)
        cv2.rectangle(canvas_img, self.detbox_list[column][1:3], self.detbox_list[column][3:5], (0, 255, 255), 2)

        qt_img=self.convert_cv_qt(canvas_img,1280,720)
        self.label_screen.setPixmap(qt_img)


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
        cv2.rectangle(bbox_img, bbox[1:3], bbox[3:5], color, thickness)
        if thickness == -1:
            bboxs_img = np.full((img.shape[0], img.shape[1], 3), (0, 0, 0), dtype=np.uint8)
            cv2.rectangle(bboxs_img, bbox[1:3], bbox[3:5], ((100-color[0])/2 ,(100-color[1])/2,(100-color[2])/2), thickness)
            img = cv2.subtract(img, bboxs_img)
        img = cv2.add(img, bbox_img)
        return img


if __name__ == "__main__":

    app = QApplication(sys.argv) 

    myWindow = WindowClass() 

    myWindow.show()

    app.exec_()