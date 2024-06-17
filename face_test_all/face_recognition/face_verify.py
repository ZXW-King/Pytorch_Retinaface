import os
import cv2
import numpy as np
import onnxruntime
import time

class Insightface():
    def __init__(self, onnxpath):
        self.onnx_session = onnxruntime.InferenceSession(onnxpath,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                           'CPUExecutionProvider'])
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    def get_input_name(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self, img_tensor):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = img_tensor
        return input_feed

    def inference(self, img_path):
        if type(img_path) == str:
            img = cv2.imread(img_path)  # 读取图片
        else:
            img = img_path
        # print(img.shape)
        or_img = cv2.resize(img, (112, 112))
        img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)
        img=img-127.5
        img /= 127.5
        img = np.expand_dims(img, axis=0)
        # print(img)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]
        return pred, or_img
    def otherinfer(self,img):
        or_img = cv2.resize(img, (112, 112))
        img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)
        img = img - 127.5
        img /= 127.5
        img = np.expand_dims(img, axis=0)
        # print(img)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]
        return pred, or_img

# L2正则化和距离计算
def face_compare(face1,face2):
    norm1=np.linalg.norm(face1)
    face1=face1/norm1
    norm2=np.linalg.norm(face2)
    face2=face2/norm2
    diff=np.subtract(face1,face2)  
    dist=np.sum(np.square(diff),1) 
    #
    #对两个矩阵进行减法运算
    return dist

def face_embeding(insightface_model,face):
    model=Insightface(insightface_model)
    embeding,face=model.otherinfer(face)
    return embeding,face

insightface_model='./webface_r50.onnx'#'./webface_r50.onnx'  './MobileFaceNet.onnx' './arcface-r18-Glint360K.onnx' 'w600k_mbf_sc.onnx' 'w600k_mbf_buffalo_s'

def compare_two_face(onnx_model_path, img_base_path, img_verify_pth):
    model=Insightface(insightface_model)
    output,re_img=model.inference(img_base_path)
    output2,imgsda=model.inference(img_verify_pth)
    distance=face_compare(output2,output)
    return distance
    
if __name__=="__main__":
    model=Insightface(insightface_model)
    
    register_files = ['./person_imgs/TCL_1.png', './person_imgs/xuwei-face.png', './person_imgs/liudehua-false.png', './person_imgs/zhangjin.png', './person_imgs/yaoming.png'] # './person_imgs/TCL_1.png'
    #register_files = ['./person_imgs/xuwei-face.png']
    for register_file in register_files:
        start_time = time.perf_counter()
        output,re_img=model.inference(register_file)
        end_time = time.perf_counter()
        print("********{} size:{} register_t:{} ms @CPU*********".format(insightface_model, output.shape, (end_time - start_time)*1000))
        max_dis = -1.0
        min_dis = 99999.0
        max_dis_name = ''
        min_dis_name = ''
        print("----------{} AS emb---------".format(register_file))
        for p in os.listdir("./know_persons_face"):
            #print(os.path.join("./know_persons_face", p))
            start_time = time.perf_counter()
            output2,imgsda=model.inference(os.path.join("./know_persons_face", p))
            dis=face_compare(output2,output)
            end_time = time.perf_counter()
            print("{} vs {}:{} verify_t:{} ms @CPU".format(register_file.split('/')[-1].split('.jpg')[0], p.split(".jpg")[0],dis,  (end_time - start_time)*1000))
            if dis < min_dis:
                min_dis = dis
                min_dis_name = p.split(".jpg")[0]
            if dis > max_dis:
                max_dis = dis 
                max_dis_name = p.split(".jpg")[0]
        print("unsimilar {}:{}, similar {}:{}".format(max_dis_name, max_dis, min_dis_name, min_dis))
    # print(output.dtype)

