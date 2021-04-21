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

from predictor import load_images, ClassificationPredictor, RatinaNetPrediction, YOLOPrediction
from association_analyzer import calculate_support, calculate_association

# --------------------------------------------------------------------------------------------------------
# ImageAnalyzer
# --------------------------------------------------------------------------------------------------------
class ImageAnalyzer:
    def __init__(   
                    self, 
                    sentiment_classifier_path:str,
                    sentiment_classifier_pre_prep_func:object,
                    sentiment_classifier_class_label:dict,

                    style_classifier_path:str,
                    style_classifier_pre_prep_func:object,
                    style_classifier_class_label:dict,

                    scence_classifier_path:str,
                    scence_classifier_pre_prep_func:object,
                    scence_classifier_class_label:dict,                    

                    object_detection_model_path:str,
                    object_detection_model_weight_path:str,
                    object_detection_class_label:dict,
                    object_detection_algorithm:str='yolo',

                    frequent_itemsets_algorithm:str='apriori',
                    min_support:float=0.3,
                    association_metric:str='confidence',
                    association_min_threshold:float=1

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

        self.__scence_cls = ClassificationPredictor( model_path=scence_classifier_path, 
                                                    preprocess_input=scence_classifier_pre_prep_func, 
                                                    class_label=scence_classifier_class_label)
        
                                    
        if object_detection_algorithm == 'yolo':
            self.__object_detection = YOLOPrediction(   model_path=object_detection_model_path,
                                                        model_weight_path=object_detection_model_weight_path,
                                                        class_label=object_detection_class_label)
        elif object_detection_algorithm == 'ratinanet':
            self.__object_detection = RatinaNetPrediction(  model_path=object_detection_model_path, 
                                                            class_label=object_detection_class_label)

        self.frequent_itemsets_algorithm = frequent_itemsets_algorithm
        self.min_support = min_support
        self.association_metric = association_metric
        self.association_min_threshold = association_min_threshold

        self.__loaded_image = False

    def __check_load_image(class_method):
        def method_wrapper(self, *arg, **kwarg):
            if self.__loaded_image:
                return class_method(self, *arg, **kwarg)
            else:
                raise Exception('You must call .load_image() first.')

        return method_wrapper

    def load_image(self, img_glob_pathname):
        # Load all image
        self.__X, self.list_of_image_path = load_images(img_glob_pathname, color_mode='rgb')
        self.__loaded_image = True

    @__check_load_image
    def sentiment_classification(self):
        predictions = self.__senti_cls.predict(self.__X)
        return self.__senti_cls.decode_predictions(predictions, top=None)

    @__check_load_image
    def style_classification(self):
        predictions = self.__style_cls.predict(self.__X)
        return self.__style_cls.decode_predictions(predictions, top=None)

    @__check_load_image
    def scence_classification(self):
        predictions = self.__style_cls.predict(self.__X)
        return self.__scence_cls.decode_predictions(predictions, top=None)

    @__check_load_image
    def frequent_object_set(self):
        
        predictions = self.__object_detection.predict(self.__X)

        decoded_predictions = self.__object_detection.decode_predictions(predictions)
        
        single_support_df = calculate_support(decoded_predictions)

        frequent_itemsets_df, association_rules_df = calculate_association(decoded_predictions, 
                                                                            frequent_itemsets_algorithm='apriori',
                                                                            min_support=self.min_support,
                                                                            association_metric=self.association_metric,
                                                                            association_min_threshold=self.association_min_threshold)

        return single_support_df, association_rules_df

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

    def summary_classification_result(self, result_df):
        idmax_result_df = result_df.iloc[:,1:].idxmax(axis="columns")
        idmax_result_df = idmax_result_df.value_counts().to_frame()
        idmax_result_df.columns = ['Number of Image']
        idmax_result_df['Percentage of Image'] = 100*idmax_result_df['Number of Image']/result_df.count()
        idmax_result_df.sort_values(by='Number of Image', ascending=False, inplace = True)
        idmax_result_df = idmax_result_df.reset_index()
        idmax_result_df.columns = ['Class', 'Number of Image', 'Percentage of Image']
        return idmax_result_df

    @__check_load_image
    def analyze(self):

        sentiment_result = self.sentiment_classification()
        style_result = self.style_classification()
        scence_result = self.scence_classification()

        # Raw classification result
        result_sentiment_df = self.create_classification_result_df(sentiment_result)
        result_style_df = self.create_classification_result_df(style_result)
        result_scence_df = self.create_classification_result_df(scence_result)

        # try:
        single_support_df, association_rules_df = self.frequent_object_set()
        # except:
        #     single_support_df = pd.DataFrame(columns=['Object', 'Number of Object', 'Number of Image', 'Support: Object', 'Support: Image'])
        #     association_rules_df = pd.DataFrame(columns=['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'])


        return  {
                    'Raw Sentiment Analysis':result_sentiment_df, 
                    'Raw Style Analysis': result_style_df, 
                    'Raw Scence Analysis': result_scence_df, 

                    'Summary Sentiment Analysis': self.summary_classification_result(result_sentiment_df),
                    'Summary Style Analysis': self.summary_classification_result(result_style_df),
                    'Summary Scence Analysis': self.summary_classification_result(result_scence_df),

                    'Support': single_support_df, 
                    'Association Rules': association_rules_df,

                    'image_path': self.list_of_image_path
                }

if __name__ == '__main__':

    pass


    