
import colorsys
import ray
import time
import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.ops import nms
import warnings

warnings.filterwarnings("ignore")

t_t = time.time()



root_path = "../model"
device = torch.device("cpu")
model = torch.jit.load(f"{root_path}/script_model.pt",map_location=device)
classes_path = f'{root_path}/voc_classes.txt'

# 模型加载完成，需要对三个框架进行适应性改造
# 模型以完成调试，已关闭无关紧要的输入I/O避免因控制台I/O影响速度
# img = Image.open("./street.jpg")


def loc2bbox(src_bbox, loc):
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_width = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_ctr_x = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height

    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

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
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
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


class FRCNN(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        # "model_path": 'model_data/voc_weights_resnet.pth',
        "classes_path": classes_path,
        # ---------------------------------------------------------------------#
        #   网络的主干特征提取网络，resnet50或者vgg
        # ---------------------------------------------------------------------#
        "backbone": "resnet50",
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   用于指定先验框的大小
        # ---------------------------------------------------------------------#
        'anchors_size': [8, 16, 32],
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化faster RCNN
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.std = self.std.cuda()
        self.bbox_util = DecodeBox(self.std, self.num_classes)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, crop=False, count=False):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------#
        #   计算resize后的图片的大小，resize后的图片短边为600
        # ---------------------------------------------------#
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#

        # ---------------------------------------------------------#
        #   给原图像进行resize，resize到短边为600的大小上
        # ---------------------------------------------------------#
        image_data = resize_image(image, [input_shape[1], input_shape[0]])
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # -------------------------------------------------------------#
            #   roi_cls_locs  建议框的调整参数
            #   roi_scores    建议框的种类得分
            #   rois          建议框的坐标
            # -------------------------------------------------------------#
            roi_cls_locs, roi_scores, rois, _ = model(images)
            # -------------------------------------------------------------#
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            # -------------------------------------------------------------#
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape,
                                             nms_iou=self.nms_iou, confidence=self.confidence)
            # ---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            # ---------------------------------------------------------#
            if len(results[0]) <= 0:
                return image

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

            pred_res = []
            for i, c in list(enumerate(top_label)):
                temp = {}
                predicted_class = self.class_names[int(c)]
                box = top_boxes[i]
                score = top_conf[i]
                temp['predicted_class'] = predicted_class
                temp['box'] = box
                temp['score'] = score
                pred_res.append(temp)

        return pred_res



# @ray.remote(num_cpus=1)
# def main():
#     ti = time.time()
#     fs = FRCNN()
#     pics = np.load('pics300.npz', allow_pickle=True)
#     imgs = pics["img"]
#     imglen = len(imgs)
#
#     for index, i in enumerate(imgs):
#
#         i = cv2.resize(i, (600, 600))
#         n_image = Image.fromarray(i)
#         res = fs.detect_image(image=n_image)
#         # if index % 30 == 0:
#         #     print(f'进度： {(index / imglen) * 100} %')
#     return ray._private.services.get_node_ip_address() + ":" + str(time.time() - ti) + ":" + str(ti)
#

@ray.remote(num_cpus=1)
def main(fs_ref, imgs_ref):
    ti = time.time()
    fs = ray.get(fs_ref)      # 获取模型
    imgs = ray.get(imgs_ref)  # 获取图片
    imglen = len(imgs)

    for index, i in enumerate(imgs):

        i = cv2.resize(i, (600, 600))
        n_image = Image.fromarray(i)
        res = fs.detect_image(image=n_image)
        # if index % 30 == 0:
        #     print(f'进度： {(index / imglen) * 100} %')
    return ray._private.services.get_node_ip_address() + ":" + str(time.time() - ti) + ":" + str(ti)


if __name__ == "__main__":
    t1 = time.time()
    ray.init(address='auto')
    fs = FRCNN()
    pics = np.load(f'{root_path}/pics300.npz', allow_pickle=True)
    imgs = pics["img"]
    # 3. 将模型和数据放入 Ray 的对象存储
    fs_ref = ray.put(fs)  # 存储模型
    imgs_ref = ray.put(imgs)  # 存储图片数据
    ips = ray.get([main.remote() for _ in range(4)])
    result = []
    for res in ips:
        node_ip, run_time, res_time = res.split(":")
        result.append({"node": node_ip, "runtime": run_time, "restime": float(res_time) - t1})
    print(result)
    print("all time: ", time.time() - t1)
    print("result: ", result)


#
#
# start = time.time()
# fs = FRCNN()
#
# pics = np.load('pics300.npz', allow_pickle=True)
# imgs = pics["img"]
# imglen = len(imgs)
#
# for index, i in enumerate(imgs):
#
#     i = cv2.resize(i, (600, 600))
#     n_image = Image.fromarray(i)
#     res = fs.detect_image(image=n_image)
#     if index % 30 == 0:
#         print(f'进度： {(index / imglen) * 100} %')
#
# end = time.time()
# print(end - start)
