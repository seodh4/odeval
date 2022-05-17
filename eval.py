import numpy as np
import os
import json
from tqdm import tqdm


class Fileio:

    def __init__(self):
        super().__init__()

    def read_file(self, path, filename):
        with open(path, 'r') as f:
            json_data = json.load(f)

            dst_shapes=[]
            for shape in json_data['shapes']:
                dst_label = shape['label']
                points = shape['points']
                
                x1 = int(points[0][0])
                y1 = int(points[0][1])
                x2 = int(points[1][0])
                y2 = int(points[1][1])


                dst_shape = [dst_label,x1,y1,x2,y2,filename]
                dst_shapes.append(dst_shape)
        return dst_shapes


        bndBoxFile = open(path, 'r')
        for bndBox in bndBoxFile:
            x1, y1, x2, y2, label= bndBox.strip().split(',')
            box.append([label, int(x1), int(y1), int(x2), int(y2)])

        return box



class Eval:

    def __init__(self):
        super().__init__()

    def IoU(self, box1, box2):
        # box = (x1, y1, x2, y2)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = round(inter / (box1_area + box2_area - inter),3)
        return iou


    def evaluation(self, gtbox_list, detbox_list, iou_th):

        gtbox_n = len(gtbox_list)
        detbox_n = len(detbox_list)

        evalbox = np.zeros(shape=(gtbox_n,detbox_n))


        # self.tableWidget.setRowCount(gtbox_n)
        # self.tableWidget.setColumnCount(detbox_n)
        # self.tableWidget.cellClicked.connect(self.cell_clicked)


        # self.tableWidget.verticalHeader().sectionClicked.connect(self.verticalHeader_clicked)
        # self.tableWidget.HorizontalHeader().sectionClicked.connect(self.HorizontalHeader_clicked)


        # self.img = cv2.imread('./gt/' + gtfile[:-5]+'.jpg')
        

        class_label = []

        for gtidx, gtbox in enumerate(gtbox_list):
            gtbox_coordinate = gtbox[1:]
            gtbox_label = gtbox[0]

            class_label.append(gtbox_label)

            for detidx, detbox in enumerate(detbox_list):
                detbox_coordinate = detbox[1:]
                detbox_label = detbox[0]

                iou = self.IoU(gtbox_coordinate,detbox_coordinate)
                evalbox[gtidx][detidx] = iou

                # item = QTableWidgetItem(str(iou))
                # self.tableWidget.setItem(gtidx, detidx, item)
                class_label.append(detbox_label)

        class_label_set = set(class_label) #집합set으로 변환
        class_label = list(class_label_set)
        

        gt_list = []
        over_gt_list = []
        e2e_gt_list = []
        fn_list = []

        tp_list = []
        e2e_tp_list = []
        fp_list = []
        overlab_list = []
        e2e_overlab_list = []

        


        for det_idx, gt_eval_list in enumerate(evalbox.T):
            # label = detbox_list[det_idx][0]
            gt_TP_idx_list = np.where(gt_eval_list >= iou_th)[0]
            # gt_TP_max_idx = np.where(gt_eval_list == max(gt_eval_list))[0]
            gt_TP_max_idx=np.argmax(gt_eval_list)
            #FP
            if len(gt_TP_idx_list) == 0:
                fp_list.append(detbox_list[det_idx])
            if len(gt_TP_idx_list) > 1:
                for idx in gt_TP_idx_list:
                    if idx == gt_TP_max_idx:
                        pass
                    else:
                        evalbox.T[det_idx][idx] = 0

       




        for gt_idx, det_eval_list in enumerate(evalbox):
            
            label = gtbox_list[gt_idx][0]
            det_TP_idx_list = np.where(det_eval_list >= iou_th)[0]

            #FN
            if len(det_TP_idx_list) == 0:
                fn_list.append(gtbox_list[gt_idx])
            else:
                pass
                # gt_list.append(gtbox_list[gt_idx])

            #overlab
            if len(det_TP_idx_list) > 1:
                over_gt_list.append(gtbox_list[gt_idx])
                for det_TP_idx in det_TP_idx_list:
                    # self.tableWidget.item(gt_idx, det_TP_idx).setForeground(QBrush(QColor(0, 0, 100)))
                    
                    if label == detbox_list[det_TP_idx][0]: 
                        e2e_overlab_list.append(detbox_list[det_TP_idx])
                        # self.tableWidget.item(gt_idx, det_TP_idx).setBackground(QBrush(QColor(180, 255, 180)))                    
                    else:
                        overlab_list.append(detbox_list[det_TP_idx])
                        # self.tableWidget.item(gt_idx, det_TP_idx).setBackground(QBrush(QColor(255, 180, 180)))
      
            #TP
            if len(det_TP_idx_list) == 1:

                det_TP_idx = det_TP_idx_list[0]
                # self.tableWidget.item(gt_idx, det_TP_idx).setForeground(QBrush(QColor(0, 100, 0)))
                if label == detbox_list[det_TP_idx][0]:
                    e2e_tp_list.append(detbox_list[det_TP_idx])
                    gtbox_list[gt_idx].append(detbox_list[det_TP_idx][0])
                    gtbox_list[gt_idx].append(evalbox[gt_idx][det_TP_idx]) # iou
                    e2e_gt_list.append(gtbox_list[gt_idx])
                    # self.tableWidget.item(gt_idx, det_TP_idx).setBackground(QBrush(QColor(180, 255, 180)))
                else:
                    tp_list.append(detbox_list[det_TP_idx])
                    gtbox_list[gt_idx].append(detbox_list[det_TP_idx][0])
                    gtbox_list[gt_idx].append(evalbox[gt_idx][det_TP_idx]) # iou
                    gt_list.append(gtbox_list[gt_idx])
                    # self.tableWidget.item(gt_idx, det_TP_idx).setBackground(QBrush(QColor(255, 180, 180)))
                    


        # print(evalbox)
        # print(fp_list)
        # print(tp_list)
        # print(e2e_tp_list)
        # print(fn_list)
        # print(overlab_list)
        # print(e2e_overlab_list)

        det_n = len(tp_list)+ len(e2e_tp_list) + len(overlab_list) + len(e2e_overlab_list) + len(fp_list)
        gt_n = len(fn_list) + len(gt_list) + len(e2e_gt_list) + len(over_gt_list)

        if gtbox_n == gt_n:
            pass
        else:
            print(evalbox)
            print("gtbox_n faild")
          
        
        if detbox_n == det_n:
            pass
        else:
            print(evalbox)
            print("detbox_n faild")
            
        det_precision =  round(((len(tp_list)+ len(e2e_tp_list)) / det_n),4)
        det_recall = round(((len(tp_list)+ len(e2e_tp_list)) / gt_n),4)
        e2e_precision =  round(((len(e2e_tp_list)) / det_n),4)
        e2e_recall = round(((len(e2e_tp_list)) / gt_n),4)
        
        resultbox = {
            "tp_box" : tp_list,
            "e2e_tp_box": e2e_tp_list,
            "fp_box" : fp_list,
            "fn_box" : fn_list,
            "gt_box" : gt_list,
            "over_gt_box" : over_gt_list,
            "e2e_gt_box": e2e_gt_list,
            "overlab_box": overlab_list,
            "e2e_overlab_box": e2e_overlab_list
        }
        eval_score = {
            "det_precision" : det_precision,
            "det_recall": det_recall,
            "e2e_precision": e2e_precision,
            "e2e_recall": e2e_recall
        }
        # evalbox2 = evalbox.tolist()
        return(resultbox , eval_score, class_label)

        