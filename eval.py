import numpy as np
import os
import json

class Fileio:

    def __init__(self):
        super().__init__()

    def read_file(self, path):
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


                dst_shape = [dst_label,x1,y1,x2,y2]
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
        iou = round(inter / (box1_area + box2_area - inter),2)
        return iou

    