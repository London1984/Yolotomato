import cv2
import numpy as np
import argparse
import time
from pathlib import Path

import torch
import collections
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class DetectionResult:
    def __init__(self):
        #(x,y,z,alpha,beta,gamma)
        self.position = list()


class TargetDetection:
    def __init__(self):
        self.weights = 'weights/tomato_last.pt'
        self.imgsz = 1920
        self.device = ''
        self.augment = False
        self.conf_thres = 0.6
        self.iou_thres = 0.45
        self.classes= None
        self.agnostic_nms= False
        self.max_det=1000
        self.project = 'runs/detect'
        self.name = 'exp'
        self.save_txt = False
        self.save_conf = False
        self.save_crop = False
        self.save_img = True
        self.hide_labels = False
        self.hide_conf = False
        self.line_thickness = 2
        self.view_img = False
        #camera parameters
        #成熟度基准
        self.scoreline = 65
        #姿态角贝塔和伽马
        self.beta = float(0)
        self.gama = float(0)
        #相机参数矩阵
        self.Camera_matrix = np.mat([[1.0978070038298756e+03, 0., 9.8276397742807512e+02], [0.,
                                                                               1.0937034731526098e+03,
                                                                               5.2343038050107725e+02], [0., 0., 1.]])
        self.inv = self.Camera_matrix.I
        #深度偏移量
        self.DepthShift = 63.6500797655807822e+00

        # Directories
        #############################图片结果保存位置###############################
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=True)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16


    # 定义色彩列表
    def getColorList(self):
        dict = collections.defaultdict(list)

        # 红色
        lower_red = np.array([156, 43, 46])
        upper_red = np.array([180, 255, 255])
        color_list = []
        color_list.append(lower_red)
        color_list.append(upper_red)
        dict['red'] = color_list

        # 红色2
        lower_red = np.array([0, 43, 46])
        upper_red = np.array([10, 255, 255])
        color_list = []
        color_list.append(lower_red)
        color_list.append(upper_red)
        dict['red2'] = color_list

        # 橙色
        lower_orange = np.array([11, 43, 46])
        upper_orange = np.array([25, 255, 255])
        color_list = []
        color_list.append(lower_orange)
        color_list.append(upper_orange)
        dict['orange'] = color_list

        return dict


    # 输入图像，针对番茄色彩阈值提取区域,计算中心和惯性主轴并返回结果
    def getTomatoPos(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_dict = self.getColorList()
        masks = []
        for d in color_dict:
            mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
            masks.append(mask)

        output = cv2.add(cv2.add(masks[0], masks[1]), masks[2])
        cv2.bitwise_not(output,output)
        cv2.imwrite('runs/detect/output.jpg',output)
        cv2.imwrite('runs/detect/cut.jpg',image)
        contours,hierarchy = cv2.findContours(output,1,1)
        
        
        # print(np.vstack(contours))
        cnt = np.vstack(contours)
        (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
        # x,y,angle = 20,20,30
       
        return x,y,angle

    #计算番茄果实成熟度
    def maturityscore(self,image):
        width,height,chanel = image.shape
        img_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        histh = cv2.calcHist([img_hsv],[0],None,[180],[0,179])
        #caculate the maturity score
        mature_pixel = histh[0]+histh[1]+histh[2]+histh[3]+histh[4]+histh[5]+histh[6]+histh[7]+histh[8]+histh[9]+histh[10]+histh[175]+histh[176]+histh[177]+histh[178]+histh[179]
        halfmature_pixel =histh[11]+histh[12]+histh[13]+histh[14]+histh[15]
        raremature_pixel = histh[16]+histh[17]+histh[18]+histh[19]+histh[20]
        total_pixel = width*height
        score = int((mature_pixel*100+halfmature_pixel*80+raremature_pixel*60)/total_pixel)
        return score

    def image_detection(self, source, depthpath) -> DetectionResult:
        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()
        
        # Set Dataloader
        vid_path, vid_writer = None, None
        
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)#read RGB image
        depth = cv2.imread(depthpath,0)#read depth image
        # if webcam:
        #     view_img = check_imshow()
        #     cudnn.benchmark = True  # set True to speed up constant image size inference
        #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        # else:
        #     dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=self.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                    max_det=self.max_det)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            result = DetectionResult()
            for i, det in enumerate(pred):  # detections per image
                # if webcam:  # batch_size >= 1
                #     p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                # else:
                #     p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for opt.save_crop
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            # print(xyxy)
                            boximage = im0s[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])] # 裁剪坐标为[y0:y1, x0:x1]
                            score = self.maturityscore(boximage)
                            # print(score)
                            label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]}Conf:{conf:.2f} M:{score}')
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=self.line_thickness)
                            if self.save_crop:
                                save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)
                           
                           #针对目标成熟度以上的果实，计算空间位置和姿态，列表形式返回结果
                            if score >self.scoreline:
                                cntx,cnty,alpha = self.getTomatoPos(boximage)
                                # print(cntx,cnty,angle)
                                imgx,imgy = int(cntx+int(xyxy[0])),int(cnty+int(xyxy[1]))
                                camera_index = np.mat([[imgx], [imgy], [1]])
                                # print(camera_index)
                                realz = depth[imgy, imgx] + self.DepthShift
                                world_index = self.inv * camera_index * (depth[imgy, imgx])
                                # print(world_index)
                                world_index.tolist()
                                # print(world_index)
                                result.position.append((float(world_index[0]), float(world_index[1]), float(realz),float(alpha),self.beta,self.gama))


                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if self.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            print(f"Results saved to {self.save_dir}{s}")
        # print(result.position)
        print(f'Done. ({time.time() - t0:.3f}s)')
        # print(result)
        return result

 

