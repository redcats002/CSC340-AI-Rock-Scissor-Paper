# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2, imutils
import time
import numpy as np
import torch
import pyshine as ps

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode



class Ui_Dialog(object):
   
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1400, 815)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 40, 1280, 720))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label.setText("")
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setPixmap(QtGui.QPixmap("robot.jpg"))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(40, 760, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(1220, 760, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.pushButton.clicked.connect(self.countdown)
        self.pushButton_2.clicked.connect(self.loadImage)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "count"))
        self.pushButton_2.setText(_translate("Dialog", "start"))

       
        # Added code here
        self.filename = 'Snapshot '+str(time.strftime("%Y-%b-%d at %H.%M.%S %p"))+'.png' # Will hold the image address location
        self.tmp = None # Will hold the temporary image for display
        self.brightness_value_now = 0 # Updated brightness value
        self.blur_value_now = 0 # Updated blur value
        self.fps=0
        self.started = False

   
    def loadImage(self):
            """ This function will load the camera device, obtain the image
                and set it to label using the setPhoto function
            """
            @smart_inference_mode()
            def runkoko(
                    weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
                    source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
                    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                    imgsz=(720, 1280),  # inference size (height, width)
                    conf_thres=0.25,  # confidence threshold
                    iou_thres=0.45,  # NMS IOU threshold
                    max_det=1000,  # maximum detections per image
                    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                    view_img=False,  # show results
                    save_txt=False,  # save results to *.txt
                    save_conf=False,  # save confidences in --save-txt labels
                    save_crop=False,  # save cropped prediction boxes
                    nosave=False,  # do not save images/videos
                    classes=None,  # filter by class: --class 0, or --class 0 2 3
                    agnostic_nms=False,  # class-agnostic NMS
                    augment=False,  # augmented inference
                    visualize=False,  # visualize features
                    update=False,  # update all models
                    project=ROOT / 'runs/detect',  # save results to project/name
                    name='exp',  # save results to project/name
                    exist_ok=False,  # existing project/name ok, do not increment
                    line_thickness=3,  # bounding box thickness (pixels)
                    hide_labels=True,  # hide labels
                    hide_conf=True,  # hide confidences
                    half=False,  # use FP16 half-precision inference
                    dnn=False,  # use OpenCV DNN for ONNX inference
                    vid_stride=1,  # video frame-rate stride
                    timeco = ""
                ):
                source = str(0)
                save_img = not nosave and not source.endswith('.txt')  # save inference images
                is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
                is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
                webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
                if is_url and is_file:
                    source = check_file(source)  # download

                # Directories
                save_dir = increment_path(Path('runs/detect') / 'exp', exist_ok=exist_ok)  # increment run
                (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

                # Load model
                device = select_device('')
                model = DetectMultiBackend('best.pt', device=device, dnn=False, data=data, fp16=False)
                stride, names, pt = model.stride, model.names, model.pt
                imgsz = check_img_size((640, 480), s=stride)  # check image size
                print(imgsz)
                # Dataloader
                if webcam:
                    view_img = check_imshow()
                    dataset = LoadStreams(source, img_size=(640, 480), stride=stride, auto=pt, vid_stride=vid_stride)
                    bs = len(dataset)  # batch_size
                else:
                    dataset = LoadImages(source, img_size=(640, 480), stride=stride, auto=pt, vid_stride=vid_stride)
                    bs = 1  # batch_size
                vid_path, vid_writer = [None] * bs, [None] * bs

                # Run inference
                model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
                seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
                for path, im, im0s, vid_cap, s in dataset:
                    with dt[0]:
                        im = torch.from_numpy(im).to(device)
                        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim

                    # Inference
                    with dt[1]:
                        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                        pred = model(im, augment=augment, visualize=visualize)

                    # NMS
                    with dt[2]:
                        pred = non_max_suppression(pred, 0.5, iou_thres, classes, agnostic_nms, max_det=max_det)

                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        objcandetect='none'
                        seen += 1
                        if webcam:  # batch_size >= 1
                            p, im0, frame = path[i], im0s[i].copy(), dataset.count
                            s += f'{i}: '
                        else:
                            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # im.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                        s = [] # print string
                        pos = []
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                               
                            

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                pos.append(xywh[0])
                                s.append(names[int(cls)])
                                
                                print(pos)
                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                    with open(f'{txt_path}.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if save_img or save_crop or view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                    
                                if save_crop:
                                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            objcandetect = s
                            print(objcandetect)
                            
                        # Stream results
                        im0 = annotator.result()
                        imc = im0.copy() if save_crop else im0  # line 164
                        im0 = cv2.resize(im0.copy(), (1280,720))

                        frame = im0
                        
                        cv2.line(frame,(640,0),(640,720),(244,244,244),20)

                        # cv2.rectangle(frame,(0,100),(100,0),(0,255,0),3)

                        # cv2.rectangle(frame,(1180,100),(1280,0),(0,255,0),3)
                        player1=""
                        player2=""
                        
                        for x in range(len(pos)) :
                            ch = pos[x]
                            if ch > 0.5 :
                                if len(objcandetect)<1 :
                                    player1=""
                                else:
                                    if len(objcandetect)<len(pos) :
                                        player1=str(objcandetect[x-1])
                                    else:
                                        player1=str(objcandetect[x])

                            
                            else :
                                if len(objcandetect)<1 :
                                    player2=""
                                else:
                                    if len(objcandetect)<len(pos) :
                                        player2=str(objcandetect[x-1])
                                    else:
                                        player2=str(objcandetect[x])
                        checkgame = ""
                        
                        if player1=="Paper" and player2=="Paper":
                      
                            checkgame="draw"
                         
                        if player1=="Scissor" and player2=="Scissor":
                            checkgame="draw"
                        if player1=="Rock" and player2=="Rock":
                            checkgame="draw"
                            
                        if player1=="Paper" and player2=="Scissor":
                            checkgame="player2-win"
                        if player1=="Scissor" and player2=="Rock":
                            checkgame="player2-win"
                        if player1=="Rock" and player2=="Paper":
                            checkgame="player2-win"
                            
                        if player1=="Paper" and player2=="Rock":
                            checkgame="player1-win"
                        if player1=="Scissor" and player2=="Paper":
                            checkgame="player1-win"
                        if player1=="Rock" and player2=="Scissor":
                            checkgame="player1-win"
                        # print('player1'+player1)    
                        # print('player2'+player2)    
                        print('checkgame'+checkgame)    
                        frame = cv2.flip(frame, 1)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame,checkgame,(530,50), font, 2,(0,0,255),4,cv2.LINE_AA)
                        cv2.putText(frame,timeco ,(530,330), font, 2,(0,0,255),4,cv2.LINE_AA)
                        cv2.putText(frame,player1,(10,680), font, 2,(0,0,255),4,cv2.LINE_AA)
                        frameready = cv2.putText(frame,player2,(1100,700), font, 2,(0,0,255),4,cv2.LINE_AA)
                        frameready = cv2.cvtColor(frameready, cv2.COLOR_BGR2RGB)
                        
                        image = QImage(frameready, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
                        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

                    self.update()
                    key = cv2.waitKey(1) & 0xFF
                    if self.started==False:
                        break
                        print('Loop break')
        
            
            if self.started:
                self.started=False
                self.pushButton_2.setText('Start')	
            else:
                self.started=True
                self.pushButton_2.setText('Stop')
                check_requirements(exclude=('tensorboard', 'thop'))
                runkoko()

    def update(self):
            """ This function will update the photo according to the 
                current values of blur and brightness and set it to photo label.
            """
            # Here we add display text to the image
            text  =  'FPS: '+str(self.fps)
            text = str(time.strftime("%H:%M %p"))

    def countdown(self):
        t=3
        while t:
            mins, secs = divmod(t, 60)
            timeco = '{:02d}:{:02d}'.format(mins, secs)
            time.sleep(1)
            t -= 1
          
        timeco = 'Fire in the hole!!'
  
  
   
        
# Subscribe to PyShine Youtube channel for more detail! 

# WEBSITE: www.pyshine.com



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
    
