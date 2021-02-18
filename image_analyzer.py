import glob
import numpy as np
import pandas as pd 
import cv2
import tensorflow
from tensorflow.keras.preprocessing import image

from predictor import ClassificationPredictor
from object_detection import CountObjectImage

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

    def analyze(self):
        sentiment_result = self.sentiment_classification()
        style_result = self.style_classification()

        sentiment_result_dict = list()
        for result in sentiment_result:
            sentiment_result_dict.append(dict(result))

        style_result_dict = list()
        for result in style_result:
            style_result_dict.append(dict(result))

        img_path_df = pd.DataFrame(self.list_of_image_path, columns=['path'])
        sentiment_result_df = pd.DataFrame(sentiment_result_dict)
        style_result_df = pd.DataFrame(style_result_dict)

        result_sentiment_df = pd.concat([img_path_df, sentiment_result_df], axis=1)
        result_style_df = pd.concat([img_path_df, style_result_df], axis=1)

        # ----------------------------------------------
        try:
            summary_frequent_object_table_df, freq_items_set_df = self.frequent_object_set()
        except:
            summary_frequent_object_table_df = pd.DataFrame(columns=['Object', 'Number of Object', 'Number of Image', 'Support: Object', 'Support: Image'])
            freq_items_set_df = pd.DataFrame(columns=['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'])


        return result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df


if __name__ == '__main__':
    ia = ImageAnalyzer(

                        sentiment_classifier_path='model/sentiment_classification.h5',
                        sentiment_classifier_pre_prep_func = tensorflow.keras.applications.efficientnet.preprocess_input,
                        sentiment_classifier_class_label=['Highly_Negative', 'Negative', 'Neutral', 'Positive', 'Highly_Positive'],

                        style_classifier_path='model/style_classification.h5',
                        style_classifier_pre_prep_func = tensorflow.keras.applications.efficientnet.preprocess_input,
                        style_classifier_class_label=[  'Bokeh','Bright','Depth_of_field','Detailed','Ethereal','Geometric_composition',
                                                            'Hazy', 'Hdr', 'Horror', 'Long_exposure', 'Macro', 'Melancholy', 'Minimal', 'Noir', 
                                                            'Pastel', 'Romantic', 'Serene', 'Sunny', 'Texture','Vintage']                     
                        )
    ia.load_image(img_glob_pathname = 'test_img/*.jpg')
    ia.load_image(img_glob_pathname = 'static/downloads/hashtag/instagram/20210107_150747_UTC_โต๋ไบร์ท/*.jpg')
    
    result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df = ia.analyze()

    print(result_sentiment_df)
    print(result_style_df)
    print(summary_frequent_object_table_df)
    print(freq_items_set_df)


    