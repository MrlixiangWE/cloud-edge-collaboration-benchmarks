import colorsys
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import ray
from torch.nn import functional as F
from torchvision.ops import nms
import os



t_t = time.time()

root_path ="./"
device = torch.device("cpu")


def loc2bbox(src_bbox, loc):
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_width   = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height  = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_ctr_x   = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y   = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height

    dx          = loc[:, 0::4]
    dy          = loc[:, 1::4]
    dw          = loc[:, 2::4]
    dh          = loc[:, 3::4]

    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox


def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width

def preprocess_input(image):
    image /= 255.0
    return image

def resize_image(image, size):
    w, h        = size
    new_image   = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

class DecodeBox():
    def __init__(self, std, num_classes):
        self.std = std
        self.num_classes = num_classes + 1

    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou=0.3, confidence=0.5):
        results = []
        bs = len(roi_cls_locs)
        # --------------------------------#
        #   batch_size, num_rois, 4
        # --------------------------------#
        rois = rois.view((bs, -1, 4))
        # ----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        # ----------------------------------------------------------------------------------------------------------------#
        for i in range(bs):
            # ----------------------------------------------------------#
            #   对回归参数进行reshape
            # ----------------------------------------------------------#
            roi_cls_loc = roi_cls_locs[i] * self.std
            # ----------------------------------------------------------#
            #   第一维度是建议框的数量，第二维度是每个种类
            #   第三维度是对应种类的调整参数
            # ----------------------------------------------------------#
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])

            # -------------------------------------------------------------#
            #   利用classifier网络的预测结果对建议框进行调整获得预测框
            #   num_rois, 4 -> num_rois, 1, 4 -> num_rois, num_classes, 4
            # -------------------------------------------------------------#
            roi = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            cls_bbox = cls_bbox.view([-1, (self.num_classes), 4])
            # -------------------------------------------------------------#
            #   对预测框进行归一化，调整到0-1之间
            # -------------------------------------------------------------#
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]

            roi_score = roi_scores[i]
            prob = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):
                # --------------------------------#
                #   取出属于该类的所有框的置信度
                #   判断是否大于门限
                # --------------------------------#
                c_confs = prob[:, c]
                c_confs_m = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    # -----------------------------------------#
                    #   取出得分高于confidence的框
                    # -----------------------------------------#
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )
                    # -----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容
                    # -----------------------------------------#
                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:, None]
                    labels = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones(
                        (len(keep), 1))
                    # -----------------------------------------#
                    #   将label、置信度、框的位置进行堆叠。
                    # -----------------------------------------#
                    c_pred = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    # 添加进result里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4]) / 2, results[-1][:, 2:4] - results[-1][:,
                                                                                                        0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        return results


@ray.remote(num_cpus=8)
def main():
    import os, sys
    root_path = os.getcwd()
    sys.path.insert(0, root_path)
    print(root_path)
    ti = time.time()

    model = torch.load("./logs/best_epoch_weights.pt", map_location=device)
    # 在每个 worker 中初始化模型

    # 获取图片数据
    pics = np.load('pics300.npz', allow_pickle=True)
    imgs = pics["img"]
    # fs = FRCNN() # 此处因为直接读了模型而爆内存，没错，报错内存超108MB，模型就是108MB

    for index, i in enumerate(imgs):
        
        print(index)
        #if index % 10 ==0 :
        #    print(f"present: {index//300}")
        i = cv2.resize(i, (600, 600))
        n_image = Image.fromarray(i)
        image_shape = np.array(np.shape(n_image)[0:2])

        input_shape = get_new_img_size(image_shape[0], image_shape[1])

        image_data = resize_image(n_image, [input_shape[1], input_shape[0]])

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).to(device)

            roi_cls_locs, roi_scores, rois, _ = model(images)

            # results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape,
            #                                  nms_iou=self.nms_iou, confidence=self.confidence)
            #
            # if len(results[0]) <= 0:
            #     return image
            #
            # top_label = np.array(results[0][:, 5], dtype='int32')
            # top_conf = results[0][:, 4]
            # top_boxes = results[0][:, :4]


    return ray._private.services.get_node_ip_address() + ":" + str(time.time() - ti) + ":" + str(ti)



if __name__ == "__main__":
    t1 = time.time()

    # 同样把图片数据也放入对象存储
    ips = ray.get( [main.remote() for _ in range(4)])
    result = []
    for res in ips:
        node_ip, run_time, res_time = res.split(":")
        result.append({"node": node_ip, "runtime": run_time, "restime": float(res_time) - t1})
    print(result)
    print("all time: ", time.time() - t1)
    print("result: ", result)
