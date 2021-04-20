# predictor.py
# -------------------------------------------------------------------------------------------------------- 
# INDEPENDENT STUDY: HASHTAG ANALYZER
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --------------------------------------------------------------------------------------------------------
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
# from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth, association_rules 

# --------------------------------------------------------------------------------------------------------
# ClassificationPredictor
# --------------------------------------------------------------------------------------------------------
class ClassificationPredictor:

    def __init__(self, 
                model:object=None, 
                model_path:str=None, 
                preprocess_input=None,  
                class_label:dict=None):


        # Initial Class
        if model is not None and model_path is not None:
            raise Exception('Please spacify either model or model_path')

        self.model = model
        self.model_path = model_path
        self.preprocess_input = preprocess_input
        self.class_label = class_label
        self.loaded_model = False

    def __class_label_tolist(self, label_dict:dict):
        label = list(label_dict.items())
        label_sorted = sorted(label, key= lambda x: int(x[0]))
        return [label[1] for label in label_sorted]

    def __load_model(self):
        if not self.loaded_model:
            if self.model_path is not None:
                self.model = load_model(self.model_path)
            self.loaded_model = True

    def load_image(self, path, color_mode='rgb'):     
        list_of_image_path = glob.glob(path)
        list_of_image_path = ['/'.join(str(Path(str_path)).split('\\')) for str_path in list_of_image_path] 

        all_input_arr = list()
        for image_path in list_of_image_path:
            image = load_img(image_path, color_mode='rgb', target_size=(224, 224))
            input_arr = img_to_array(image)
            if color_mode == 'bgr':
                input_arr = input_arr[:, :, ::-1]
            #input_arr = np.expand_dims(input_arr, axis=0)
            all_input_arr.append(input_arr)
        #input_arrs = np.vstack(all_input_arr)
        input_arrs = np.stack(all_input_arr)
        return input_arrs, list_of_image_path

    def predict(self, X):

        self.__load_model()

        if self.model is None:
            raise Exception('Model not found.')
        if self.preprocess_input is not None:
            X = self.preprocess_input(X)
        predictions = self.model.predict(X)
        return predictions

    def predict_form_path(self, path):
        X, list_of_image_path = self.load_image(path)
        predictions = self.predict(X)
        return predictions, list_of_image_path

    def decode_predictions(self, predictions, top=None):
        
        decoded = list()
        class_label = self.class_label.copy()

        if class_label is None:
            class_label = range(len(predictions[0]))
        else:
            class_label = self.__class_label_tolist(class_label)

        for prediction in predictions:

            result = list((zip(class_label, list(prediction))))
            if top is not None:
                result = sorted(result, key=lambda i: i[1], reverse=True)[:top]
            decoded.append(result)

        return decoded

# --------------------------------------------------------------------------------------------------------
# RatinaNetPrediction
# --------------------------------------------------------------------------------------------------------

class RatinaNetPrediction(ClassificationPredictor):

    def __init__(self, 
                model_path:str=None, 
                class_label:dict=None):

        ClassificationPredictor.__init__(self, None, model_path, None, class_label)

    def __load_model(self):
        if not self.loaded_model:
            if self.model_path is not None:
                self.model = models.load_model(self.model_path, backbone_name='resnet50')
            self.loaded_model = True
        
    def load_image(self, path, color_mode='bgr'): 
        return ClassificationPredictor.load_image(self, path, color_mode)

    def predict(self, X):

        self.__load_model()

        if self.model is None:
            raise Exception('Model not found.')

        images = list()
        for image in X:
            image, scale = resize_image(image)
            #image = np.expand_dims(image, axis=0)
            images.append(image)

        #X = np.vstack(images)
        X = np.stack(images)
        X = preprocess_image(X)

        predictions = self.model.predict_on_batch(X)
        predictions[0] = predictions[0].astype('int')

        return predictions

    def predict_form_path(self, path):
        X, list_of_image_path = self.load_image(path)
        predictions = self.predict(X)
        return predictions, list_of_image_path

    def decode_prediction(self, prediction, confident_threshold:float=0.5, non_maxium_suppression_threshold:float=0.3):

        boxes, scores, class_ids = prediction

        selected = class_ids > -1
        exist_class_ids = class_ids[selected]
        exist_scores = scores[selected]
        exist_boxes = list(boxes[selected])

        qualify_index = cv2.dnn.NMSBoxes(exist_boxes, exist_scores, confident_threshold, non_maxium_suppression_threshold)

        if len(qualify_index) != 0:
            qualify_index = tuple(qualify_index.flatten())

        result_df = pd.DataFrame(columns = ['class_id', 'score', 'box'])
        result_df['class_id'] = exist_class_ids
        result_df['score'] = exist_scores
        result_df['box'] = exist_boxes
        result_df = result_df.astype({'class_id': 'str'})
        result_df['class'] = result_df['class_id'].map(self.class_label)
        result_df = result_df.loc[qualify_index, :]

        return result_df

    def decode_predictions(self, predictions, confident_threshold:float=0.5, non_maxium_suppression_threshold:float=0.3):
        
        decoded = list()

        for i in range(len(predictions[0])):
            prediction = (predictions[0][i],  predictions[1][i],  predictions[2][i])
            prediction_df = self.decode_prediction(prediction, confident_threshold, non_maxium_suppression_threshold)
            selected_prediction_df = prediction_df[prediction_df['score'] > confident_threshold]['class']
            decoded.append(tuple(selected_prediction_df))
            
        return decoded

# --------------------------------------------------------------------------------------------------------
# YOLOPrediction
# --------------------------------------------------------------------------------------------------------

class YOLOPrediction(RatinaNetPrediction):

    def __init__(self, 
                model_path:str, # "model/YOLO-COCO/yolov4.cfg",
                model_weight_path:str, # "model/YOLO-COCO/yolov4.weights"
                class_label:dict=None):

        self.model_weight_path = model_weight_path
        self.yolo_img_width = 416             # width of the network input image
        self.yolo_img_hight = 416             # height of the network input image

        RatinaNetPrediction.__init__(self, model_path, class_label)

    def __load_model(self):
        if not self.loaded_model:

            # Load a pre-trained YOLOv3 model from disk
            self.model = cv2.dnn.readNetFromDarknet(self.model_path,self.model_weight_path)
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)        

            # Determine only the output layer names that we need from YOLO
            out_layer_name = self.model.getLayerNames()
            self.out_layer_name = [ out_layer_name[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]

            self.loaded_model = True
        
    def predict(self, X):

        self.__load_model()

        if self.model is None:
            raise Exception('Model not found.')

        all_boxes = list()
        all_scores = list()
        all_class_ids = list()

        for image in X:

            # Create a 4D blob from a frame a a preprocessing step
            # This function includes options to do
            # - mean subtraction
            # - resize or scale values by scalefactor
            # - crop (from center)
            # - swap blue and red channels

            (h,w) = image.shape[:2]
            pre_preocessed_image = cv2.dnn.blobFromImage( image,
                                            1 / 255.0, # scaleFactor
                                            (self.yolo_img_width, self.yolo_img_hight), # spatial size of the CNN
                                            swapRB=True, crop=False)

            # Pass the blob to the network
            self.model.setInput(pre_preocessed_image)
            raw_prediction = self.model.forward(self.out_layer_name)

            boxes, scores, class_ids = self.__reformat_raw_prediction(raw_prediction, h, w)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_class_ids.append(class_ids)

        return all_boxes, all_scores, all_class_ids

    def __reformat_raw_prediction(self, raw_prediction, h, w):
        

        # Lists to store detected bounding boxes, confidences and class_ids

        boxes = list()
        scores = list()
        class_ids = list()

        # Loop over each of the layer outputs
        for output in raw_prediction:
            # Loop over each of the detections
            for detection in output:
                # Extract the score (i.e., probability) and class_id
                class_prob = detection[5:]
                class_id = np.argmax(class_prob)
                score = class_prob[class_id]

                # Filter out weak detections by ensuring the score is greater than the threshold
                if score > 0:

                    # Compute the (x, y)-coordinates of the bounding box
                    box = detection[0:4] * np.array( [w,h,w,h] )
                    (centerX, centerY, width, height) = box.astype('int')
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # Add a new bounding box to our list
                    
                    # if x < 0:
                    #     x = 0
                    # if y < 0:
                    #     x = 0
                    # if width < 0:
                    #     width = 0
                    # if height < 0:
                    #     height = 0                   
                    boxes.append([x, y, int(width), int(height)])
                    scores.append(float(score))
                    class_ids.append(class_id)
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)

        return boxes, scores, class_ids

if __name__ == '__main__':
    pass




