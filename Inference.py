import tensorflow as tf
import cv2
import numpy as np

class Infer:
    def __init__(self,model_dir):
        self.model = self.load_model(model_dir)
        self.cls_lst = np.array(['background', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'plus', 'minus', 'div', 'times', 'equal'])
        self.thickness=2
        self.color=(255,0,255)
        self.font=cv2.FONT_HERSHEY_SIMPLEX


    def read_image(self,file):
        
        img2disp = cv2.imread(file)
        im2proc = tf.io.read_file(file)
        im2proc = tf.image.decode_image(im2proc)
        im2proc = tf.image.resize(im2proc,(640,640))
        return (im2proc,img2disp)


    def load_model(self,model_dir):
        model = tf.saved_model.load(model_dir)
        nm = model.signatures['serving_default']
        return nm


    def predict(self,file,score):
        im2proc,im2disp = self.read_image(file)
        height = im2disp.shape[0]
        width = im2disp.shape[1]
        detections = self.model(tf.cast(tf.expand_dims(im2proc,axis=0),dtype=tf.uint8))
        tr = detections['detection_scores']>score
        det_scors = detections['detection_scores'][tr]
        det_boxes = detections['detection_boxes'][tr]*[height,width,height,width]
        det_cls = self.cls_lst[detections['detection_classes'][tr].numpy().astype('int32')]
        for box,scr,cls in zip(det_boxes,det_scors,det_cls):
            box = list(map(int,box.numpy()))
            scr = scr.numpy()*100
            im2disp = cv2.rectangle(im2disp,(box[1],box[0]),(box[3],box[2]),thickness=self.thickness,color=self.color)
            im2disp = cv2.putText(im2disp,text=f"{cls} {scr:.2f}",org=(box[1],box[0]-4),thickness=self.thickness-1,color=self.color,fontScale=0.2,fontFace=self.font)

        cv2.imshow('Prediction',im2disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




