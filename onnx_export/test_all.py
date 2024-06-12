from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from onnx_export.file import Walk
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import onnxruntime as ort

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='FaceDetector_padding.onnx',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_image_path', type=str, default="/media/xin/data/data/face_data/test_data/res/screen-result_onnx", help='show detection results')
parser.add_argument('--display', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--imgdir', default="/media/xin/data/data/face_data/images", type=str,
                    help='visualization_threshold')
parser.add_argument('--imgsize', default=640, type=int, help='visualization_threshold')
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
    # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
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
    print('Finished loading model!')
    return loc, conf, landms


def draw_image(dets, image):
    for b in dets:
        if b[4] < args.vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)
    return image


class VideoStreamer(object):
    def __init__(self, basedir, file_path,rotate_angle=0):
        self.listing = []
        self.basedir = basedir
        self.file_path = file_path
        self.rotate_angle = rotate_angle
        self.i = 0
        if self.basedir == "camera":
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
        if self.basedir == "camera":
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
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    img_size = args.imgsize
    model_path = args.trained_model
    display = args.display
    # save_image_path = args.save_image_path
    save_image_path = ""
    # basedir = "camera"
    basedir = "cameras"
    # file_path = "/media/xin/data/data/face_data/test_img"
    file_path = "/media/xin/data/data/face_data/our_test/images"
    # file_path = "/media/xin/data/data/face_data/test_data/FairPhone5广角采集预览录屏/室内灯光/0.2m screen-20240417-144619.mp4"
    vs = VideoStreamer(basedir, file_path,0)
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
        draw_image(dets, img_raw)


        # save image
        # name = "test_padding.jpg"
        # save_path = "/media/xin/data/data/face_data/test_img_res"
        # imgname = "/".join(image_path.split("/")[-2:])
        # img_save_path = os.path.join(save_path,imgname)

        if save_image_path:
            if not os.path.exists(os.path.dirname(save_image_path)):
                os.makedirs(os.path.dirname(save_image_path))
            img_name = f"frame_{count:04d}.jpg"
            save_img = os.path.join(save_image_path,img_name)
            cv2.imwrite(save_img, img_raw)
            print("保存成功：",img_name)
            count += 1

        if display:
            img = cv2.resize(img_raw,(img_raw.shape[1]//2,img_raw.shape[0]//2))
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
