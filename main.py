import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import qdarkstyle
from eval import *
import numpy as np


from PyQt5.QtCore import pyqtSlot, Qt 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *





form_class = uic.loadUiType("windows.ui")[0]

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        iou_th = 0.2

        fileio = Fileio()
        eval = Eval()

        gtbox_list = []
        detbox_list = []

        # gtfile_list =os.listdir('./gt/')
        # gtfile_list = [file for file in gtfile_list if file.endswith(".txt")]
        # for gtfile in gtfile_list:
        #     detfile = gtfile
        gtfile = 'ICDAR03_64.txt'
        detfile = 'ICDAR03_64.txt'
        gtbox_list = fileio.read_file('./gt/'+ gtfile)
        detbox_list = fileio.read_file('./det/'+ detfile)

        gtbox_n = len(gtbox_list)
        detbox_n = len(detbox_list)

        evalbox = np.zeros(shape=(gtbox_n,detbox_n))


        self.tableWidget.setRowCount(gtbox_n)
        self.tableWidget.setColumnCount(detbox_n)
        self.tableWidget.cellClicked.connect(self.set_label)


        for gtidx, gtbox in enumerate(gtbox_list):
            gtbox_coordinate = gtbox[1:]
            gtbox_label = gtbox[0]

            for detidx, detbox in enumerate(detbox_list):
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


        tp_list = []
        fp_list = []
        fn_list = []
        overlab_list = []

        for gt_idx, gt in enumerate(evalbox):
            # print(len(np.where(gt > iou_th)[0]))
            
            gtbox_list[gt_idx] = ''

            if len(np.where(gt > iou_th)[0]) > 1: #해당 GT에 2개 이상 det 물림
                np.where(gt > iou_th)[0]
                pass
            
            print(gtbox_list)

            len_gtbox_list=len([x for x in gtbox_list if x != ''])
            print(len_gtbox_list)


            # det_eval_list = [x for x in gt if x > iou_th]
            # maxiou = max(det_eval_list)
            # maxiou_idx = np.where(gt == maxiou)[0][0]



            # print(maxiou_idx)
            # print(detbox_list[maxiou_idx])


            # tp_idx_list = np.where(gt > 0.5)
            # print(tp_idx_list)
            # print(gtbox_list[tp_idx_list[0]])


            # det_tp_list = [x for x in gt if x > 0.5]
            # print(det_tp_list)


            
            # a = np.where(gt > 0.5)
            # print(a)

                # if iou > 0.5:
                #     pass
                #     print(gtbox_label, detbox_label)



        # print(det_evalbox)
                    
           
    def set_label(self, row, column):
        item = self.tableWidget.item(row, column)
        value = item.text()
        label_string = 'Row: ' + str(row+1) + ', Column: ' + str(column+1) + ', Value: ' + str(value)
        print(label_string)
        # self.label.setText(label_string)




if __name__ == "__main__":

    app = QApplication(sys.argv) 

    myWindow = WindowClass() 

    myWindow.show()

    app.exec_()