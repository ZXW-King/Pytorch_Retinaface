from __future__ import print_function
import os
import argparse
import torch
import numpy as np
from face_recognition.face_verify import Insightface, face_compare

from file import Walk
import cv2
import onnxruntime as ort
from itertools import product as product
from math import ceil


parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('--face_detect_model', default='FaceDetector.onnx',
                    type=str, help='FaceDetector onnx model path')
parser.add_argument('--face_recognition_model', default='FaceRecognition.onnx',
                    type=str, help='FaceRecognition onnx model path')
parser.add_argument('--iscamera', action="store_true", default=False, help='Is it a video file')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_image_path', type=str, default="", help='show detection results')
parser.add_argument('--display', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--imgsize', default=640, type=int, help='visualization_threshold')
parser.add_argument('--test_path', default="", type=str, help='test path')
parser.add_argument('--register_face_image_path', default="", type=str, help='test path')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def load_model_ort(model_path):
    ort_session = ort.InferenceSession(model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                              'CPUExecutionProvider']
                                       )

    return ort_session


def padding_resize(image, target_size, padding_color=(0, 0, 0)):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate the scale ratio
    ratio_w = target_width / original_width
    ratio_h = target_height / original_height
    ratio = min(ratio_w, ratio_h)

    # Calculate new size and padding
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    pad_width = target_width - new_width
    pad_height = target_height - new_height

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Pad the image
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return padded_image, ratio, (left, top, right, bottom)


def preprocess(img_raw, img_size):
    target_size = (img_size, img_size)  # Example target size
    padding_color = (0, 0, 0)  # Example padding color (black)
    img, ratio, padding = padding_resize(img_raw, target_size, padding_color)
    img = np.float32(img)
    im_height, im_width, _ = img.shape
    img -= (104, 117, 123)
    box_scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    landms_scale = np.array([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                             img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                             img.shape[3], img.shape[2]])
    return img, box_scale, landms_scale,padding,ratio

class PriorBox(object):
    def __init__(self, cfg, format:str="tensor", image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"
        self.__format = format

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        a = np.array(anchors).reshape(-1,4)
        # back to torch land
        if self.__format == "tensor":
            output = torch.Tensor(anchors).view(-1, 4)
        elif self.__format == "numpy":
            output = np.array(anchors).reshape(-1, 4)
        else:
            raise "ERROR: INVALID TYPE OF FORMAT"

        if self.clip:
            if self.__format == "tensor":
                output.clamp_(max=1, min=0)
            else:
                output = np.clip(output, 0, 1)

        return output


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = None
    if isinstance(loc, np.ndarray) and isinstance(priors, np.ndarray):
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)

    else:
        print(type(loc), type(priors))
        print(TypeError("ERROR: INVALID TYPE OF BOUNDING BOX"))

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = None
    if isinstance(pre, np.ndarray) and isinstance(priors, np.ndarray):
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                                 ), axis=1)

    else:
        print(TypeError("ERROR: INVALID TYPE OF LANDMARKS"))

    return landms


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def postprocess(im_height, im_width, model_out, box_scale, landms_scale, padding, ratio, resize=1):
    priorbox = PriorBox(cfg, image_size=(im_height, im_width), format="numpy")
    priors = priorbox.forward()

    prior_data = priors

    boxes = decode(np.squeeze(model_out[0], axis=0), prior_data, cfg['variance'])
    boxes = boxes * box_scale / resize
    scores = np.squeeze(model_out[1], axis=0)[:, 1]

    landms = decode_landm(np.squeeze(model_out[2].data, axis=0), prior_data, cfg['variance'])

    landms = landms * landms_scale / resize

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    # Map boxes and landmarks back to original image size
    dets[:, [0, 2]] = (dets[:, [0, 2]] - padding[0]) / ratio
    dets[:, [1, 3]] = (dets[:, [1, 3]] - padding[1]) / ratio
    dets[:, 5::2] = (dets[:, 5::2] - padding[0]) / ratio
    dets[:, 6::2] = (dets[:, 6::2] - padding[1]) / ratio
    return dets


def model_run(model_path,img):
    inputs = {"input": img}
    # load onnx model
    retinaface = load_model_ort(model_path)
    loc, conf, landms = retinaface.run(None, inputs)
    return loc, conf, landms


def draw_image(dets, image):
    face_dict = {}
    count = 1
    for b in dets:
        if b[4] < args.vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        cropped_image = image[b[1]:b[3], b[0]:b[2]]
        face_dict[count] = [cropped_image,(b[0],b[1])]
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        cv2.putText(image, str(count), (cx+10, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255))

        # landms
        cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)
        count += 1
    return image,face_dict


def get_face(img_raw,img_size):
    cropped_image_list = []
    img, box_scale, landms_scale, padding, ratio = preprocess(img_raw, img_size)
    model_out = model_run(model_path, img)
    dets = postprocess(img_size, img_size, model_out, box_scale, landms_scale, padding, ratio, resize=1)
    for b in dets:
        if b[4] < args.vis_thres:
            continue
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        cropped_image = img_raw[b[1]:b[3], b[0]:b[2]]
        cropped_image_list.append(cropped_image)
    return cropped_image_list

def register_face(img_path,insightface,img_size):
    files = Walk(img_path, ['jpg', 'jpeg', 'png'])
    register_face_list = []
    for image_path in files:
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        cropped_image_list = get_face(img_raw,img_size)
        for face in cropped_image_list:
            output,re_img = insightface.inference(face)
            register_face_list.append([output,re_img])
            cv2.imshow("s",re_img)
            cv2.waitKey()
    print(f"注册的人脸个数为{len(register_face_list)}")
    return register_face_list


class VideoStreamer(object):
    def __init__(self, basedir, file_path,rotate_angle=0):
        self.listing = []
        self.basedir = basedir
        self.file_path = file_path
        self.rotate_angle = rotate_angle
        self.i = 0
        if self.basedir:
            # 打开视频文件
            self.cap = cv2.VideoCapture(self.file_path)
            # 检查视频是否成功打开
            if not self.cap.isOpened():
                raise "Error: Could not open video."
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            files = Walk(self.file_path, ['jpg', 'jpeg', 'png'])
            self.listing = files

    def read_image(self, image_path):
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return img_raw

    def next_frame(self):
        if self.basedir:
            ret, input_image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
                return
            # 如果需要旋转图像
            if self.rotate_angle != 0:
                input_image = rotate_image(input_image, self.rotate_angle)
        else:
            input_image = self.read_image(self.listing[self.i])
        self.i += 1
        return input_image

def rotate_image(image, angle):
    """
    旋转图像。

    参数：
    image (ndarray): 输入图像。
    angle (float): 旋转角度。

    返回值：
    ndarray: 旋转后的图像。
    """
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    # 计算图像中心
    center = (w / 2, h / 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后的新边界框
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵以考虑平移
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 执行实际的旋转并返回图像
    rotated = cv2.warpAffine(image, M, (new_w, new_h))

    return rotated

if __name__ == '__main__':
    cfg = {
        'name': 'mobilenet0.25',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 32,
        'ngpu': 1,
        'epoch': 250,
        'decay1': 190,
        'decay2': 220,
        'image_size': 640,
        'pretrain': True,
        'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        'in_channel': 32,
        'out_channel': 64
    }
    torch.set_grad_enabled(False)

    img_size = args.imgsize
    model_path = args.face_detect_model
    display = args.display
    save_image_path = args.save_image_path
    basedir = args.iscamera
    file_path = args.test_path
    vs = VideoStreamer(basedir, file_path,90)
    insightface = Insightface(args.face_recognition_model)
    register_face_list = None
    if args.register_face_image_path:
        register_face_list = register_face(args.register_face_image_path, insightface, img_size)
    # testing begin
    key = 32
    count = 0
    while True:
        try:
            img_raw = vs.next_frame()
        except:
            break
        img, box_scale, landms_scale,padding,ratio = preprocess(img_raw, img_size)
        model_out = model_run(model_path,img)
        dets = postprocess(img_size, img_size, model_out, box_scale, landms_scale, padding, ratio, resize=1)
        res_image,faces_dict = draw_image(dets, img_raw)
        if register_face_list:
            i = 1
            for register_face in register_face_list:
                for c,face in faces_dict.items():
                    try:
                        out,face_resize =  insightface.inference(face[0])
                        distance = face_compare(out, register_face[0])
                        print(f"注册的第{i}个人脸与图像中第{c}个人脸的距离为{distance}")
                    except:
                        continue
                i += 1

        if save_image_path:
            if not os.path.exists(os.path.dirname(save_image_path)):
                os.makedirs(os.path.dirname(save_image_path))
            img_name = f"frame_{count:04d}.jpg"
            save_img = os.path.join(save_image_path,img_name)
            cv2.imwrite(save_img, res_image)
            print("保存成功：",img_name)
            count += 1

        if display:
            img = cv2.resize(res_image,(res_image.shape[1]//2,res_image.shape[0]//2))
            cv2.imshow("test", img)
            if key == 27:
                print('Quitting, \'q\' pressed.')
                break
            if key == 32:
                while True:
                    key = cv2.waitKey(1)
                    if key == 32:
                        break
            key = cv2.waitKey(1)
    cv2.destroyAllWindows()
