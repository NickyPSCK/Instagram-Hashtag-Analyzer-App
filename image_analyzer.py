# image_analyzer.py
# -------------------------------------------------------------------------------------------------------- 
# INDEPENDENT STUDY: HASHTAG ANALYZER
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------
import glob
import numpy as np
import pandas as pd 
import cv2
import tensorflow
from tensorflow.keras.preprocessing import image

from predictor import ClassificationPredictor
from object_detection import CountObjectImage

# --------------------------------------------------------------------------------------------------------
# ImageAnalyzer
# --------------------------------------------------------------------------------------------------------
class ImageAnalyzer:
    def __init__(   
                    self, 
                    sentiment_classifier_path:str,
                    sentiment_classifier_pre_prep_func:object,
                    sentiment_classifier_class_label:list,

                    style_classifier_path:str,
                    style_classifier_pre_prep_func:object,
                    style_classifier_class_label:list
                ):

        self.__X = None
        self.list_of_image_path = None

        # Load all model
        self.__senti_cls = ClassificationPredictor( model_path=sentiment_classifier_path, 
                                                    preprocess_input=sentiment_classifier_pre_prep_func, 
                                                    class_label=sentiment_classifier_class_label)

        self.__style_cls = ClassificationPredictor( model_path=style_classifier_path, 
                                                    preprocess_input=style_classifier_pre_prep_func, 
                                                    class_label=style_classifier_class_label)
    def load_image(self, img_glob_pathname):
        # Load all image
        self.__X, self.list_of_image_path = ClassificationPredictor().load_image(img_glob_pathname)

    def sentiment_classification(self):
        predictions = self.__senti_cls.predict(self.__X)
        return self.__senti_cls.decode_predictions(predictions, top=None)

    def style_classification(self):
        predictions = self.__style_cls.predict(self.__X)
        return self.__style_cls.decode_predictions(predictions, top=None)

    def frequent_object_set(self):
        # Load labels of model
        labels = open("./model/YOLO-COCO/coco.names").read().strip().split("\n")

        # Load a pre-trained YOLOv3 model from disk
        net = cv2.dnn.readNetFromDarknet("./model/YOLO-COCO/yolov3.cfg","./model/YOLO-COCO/yolov3.weights")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 
        imgs = self.__X.copy() #[ cv2.imread(i) for i in self.list_of_image_path ]

        co = CountObjectImage(net = net, labels = labels)
        result = co.fit_predict(imgs = imgs)
        summary_table = co.summary_table()
        freq_items_set = co.get_freq_items(min_support = 0.3)

        return summary_table, freq_items_set

    def create_classification_result_df(self, classification_result):

        classification_result_dict = list()
        for result in classification_result:
            classification_result_dict.append(dict(result))

        result_df = pd.DataFrame(classification_result_dict)
        result_df['path'] = self.list_of_image_path

        path = result_df['path']

        result_df = result_df.drop(labels=['path'], axis=1)
        result_df.insert(0, 'path', path)

        return result_df


    def analyze(self):

        sentiment_result = self.sentiment_classification()
        style_result = self.style_classification()

        # Raw classification result
        result_sentiment_df = self.create_classification_result_df(sentiment_result)
        result_style_df = self.create_classification_result_df(style_result)
        
        # 

        # ----------------------------------------------
        try:
            summary_frequent_object_table_df, freq_items_set_df = self.frequent_object_set()
        except:
            summary_frequent_object_table_df = pd.DataFrame(columns=['Object', 'Number of Object', 'Number of Image', 'Support: Object', 'Support: Image'])
            freq_items_set_df = pd.DataFrame(columns=['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'])


        return result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df


if __name__ == '__main__':

    pass


    