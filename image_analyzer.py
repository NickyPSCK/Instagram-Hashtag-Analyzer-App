# image_analyzer.py
# -------------------------------------------------------------------------------------------------------- 
# INDEPENDENT STUDY: HASHTAG ANALYZER
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------
import glob
import math
import numpy as np
import pandas as pd 
import cv2
import tensorflow
from tensorflow.keras.preprocessing import image

from predictor import load_images, get_imgs_properties, ClassificationPredictor, RatinaNetPrediction, YOLOPrediction
from association_analyzer import calculate_support, calculate_association, check_objects_from_df
from util.utility import round_df

# --------------------------------------------------------------------------------------------------------
# ImageAnalyzer Utility Functions
# --------------------------------------------------------------------------------------------------------
def summary_classification_result(result_df):
    predicted_class = result_df.iloc[:,1:].idxmax(axis="columns")
    idmax_result_df = predicted_class.value_counts().to_frame()
    idmax_result_df.columns = ['Number of Image']
    idmax_result_df['Percentage of Image'] = 100*idmax_result_df['Number of Image']/result_df.count()
    idmax_result_df.sort_values(by='Number of Image', ascending=False, inplace = True)
    idmax_result_df = idmax_result_df.reset_index()
    idmax_result_df.columns = ['Class', 'Number of Image', 'Percentage of Image']
    return predicted_class, idmax_result_df

def summary_single_view(single_view_df, key:list, decimals:int=2):
    selected_col = ['path'] + key
    summary_df = single_view_df[selected_col].groupby(key).count().reset_index()
    summary_df['%'] = 100*summary_df['path']/summary_df['path'].sum()
    summary_df = summary_df.rename(columns = {'path': 'count'})
    summary_df = summary_df.sort_values(by='count', ascending=False)
    return round_df(summary_df, decimals=decimals)

def summary_check_objects(check_df, key:list=None, column_prefix:str=None, decimals:int=2):
    if key is None:
        key = list()

    if column_prefix is not None:
        prefix_key = [col_name for col_name in list(check_df.columns) if col_name.startswith(column_prefix)]
    else:
        prefix_key = list()

    check_df = check_df[key + prefix_key]
    no_of_basket = len(check_df)
    summary_df = check_df.sum().to_frame(name='count')
    summary_df['%'] = (summary_df['count']/no_of_basket) * 100
    summary_df.insert(loc=0, column='object', value=summary_df.index)
    summary_df = summary_df.reset_index(drop=True)
    return round_df(summary_df, decimals=decimals)
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

                    scene_classifier_path:str,
                    scene_classifier_pre_prep_func:object,
                    scene_classifier_class_label:dict,    
                    scene_classifier_cat_label:dict,               

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
                                                    class_label=None)

        self.__style_cls = ClassificationPredictor( model_path=style_classifier_path, 
                                                    preprocess_input=style_classifier_pre_prep_func, 
                                                    class_label=None)

        self.__scene_cls = ClassificationPredictor( model_path=scene_classifier_path, 
                                                    preprocess_input=scene_classifier_pre_prep_func, 
                                                    class_label=None)
        
                                    
        if object_detection_algorithm == 'yolo':
            self.__object_detection = YOLOPrediction(   model_path=object_detection_model_path,
                                                        model_weight_path=object_detection_model_weight_path,
                                                        class_label=None)
        elif object_detection_algorithm == 'ratinanet':
            self.__object_detection = RatinaNetPrediction(  model_path=object_detection_model_path, 
                                                            class_label=None)

        self.frequent_itemsets_algorithm = frequent_itemsets_algorithm
        self.min_support = min_support
        self.association_metric = association_metric
        self.association_min_threshold = association_min_threshold

        self.__loaded_image = False

        self.sentiment_classifier_class_label = sentiment_classifier_class_label
        self.style_classifier_class_label = style_classifier_class_label
        self.scene_classifier_class_label = scene_classifier_class_label
        self.scene_classifier_cat_label = scene_classifier_cat_label
        self.object_detection_class_label = object_detection_class_label

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
    def images_properties(self):
        result_df = get_imgs_properties(self.__X)
        result_df['path'] = self.list_of_image_path
        path = result_df['path']
        result_df = result_df.drop(labels=['path'], axis=1)
        result_df.insert(0, 'path', path)

        return result_df

    @__check_load_image
    def sentiment_classification(self):
        predictions = self.__senti_cls.predict(self.__X)
        return self.__senti_cls.decode_predictions(predictions, 
                                                    class_label=self.sentiment_classifier_class_label,
                                                    top=None)

    @__check_load_image
    def style_classification(self):
        predictions = self.__style_cls.predict(self.__X)
        return self.__style_cls.decode_predictions(predictions, 
                                                    class_label=self.style_classifier_class_label,
                                                    top=None)

    @__check_load_image
    def scene_classification(self):
        predictions = self.__scene_cls.predict(self.__X)

        result_scene_df = self.__scene_cls.decode_predictions(predictions, 
                                                                class_label=self.scene_classifier_class_label,
                                                                top=None)

        result_scene_cat_df = self.__scene_cls.decode_predictions(predictions, 
                                                                class_label=self.scene_classifier_cat_label,
                                                                top=None)                                                                
        return result_scene_df, result_scene_cat_df

    @__check_load_image
    def object_detection(self, confident_threshold:float=0.5, non_maxium_suppression_threshold:float=0.3):
        predictions = self.__object_detection.predict(self.__X)
        return self.__object_detection.decode_predictions(predictions, 
                                                                class_label=self.object_detection_class_label,
                                                                confident_threshold=confident_threshold, 
                                                                non_maxium_suppression_threshold=non_maxium_suppression_threshold)

    def frequent_object_set(self, decoded_predictions):
        
        single_support_df = calculate_support(decoded_predictions)

        frequent_itemsets_df, association_rules_df = calculate_association(decoded_predictions, 
                                                                            frequent_itemsets_algorithm='apriori',
                                                                            min_support=self.min_support,
                                                                            association_metric=self.association_metric,
                                                                            association_min_threshold=self.association_min_threshold)

        return single_support_df, association_rules_df

    def create_classification_result_df(self, classification_result, duplicate_class=True):

        classification_result_dict = list()
        for result in classification_result:

            if duplicate_class:

                dedup_result = dict()
                for each_prob in result:
                    if each_prob[0] in dedup_result:
                        dedup_result[each_prob[0]] += each_prob[1]
                    else:
                        dedup_result[each_prob[0]] = each_prob[1]
                result = dedup_result

            classification_result_dict.append(dict(result))

        result_df = pd.DataFrame(classification_result_dict)
        result_df['path'] = self.list_of_image_path

        path = result_df['path']

        result_df = result_df.drop(labels=['path'], axis=1)
        result_df.insert(0, 'path', path)

        return result_df

    @__check_load_image
    def analyze(self, tracked_objs:list=None, confident_threshold:float=0.5, non_maxium_suppression_threshold:float=0.3):

        if tracked_objs is None:
            tracked_objs = list()

        sentiment_result = self.sentiment_classification()
        style_result = self.style_classification()
        scene_result, scene_cat_result = self.scene_classification()

        # Raw classification result
        result_sentiment_df = self.create_classification_result_df(sentiment_result)
        result_style_df = self.create_classification_result_df(style_result)
        result_scene_df = self.create_classification_result_df(scene_result)
        result_scene_cat_df = self.create_classification_result_df(scene_cat_result)

        predicted_sentiment_class = result_sentiment_df.iloc[:,1:].idxmax(axis="columns")
        predicted_style_class = result_style_df.iloc[:,1:].idxmax(axis="columns")
        predicted_scence_class = result_scene_df.iloc[:,1:].idxmax(axis="columns")
        predicted_scene_cat_class = result_scene_cat_df.iloc[:,1:].idxmax(axis="columns")


        object_detection_decoded_predictions = self.object_detection(   confident_threshold=confident_threshold, 
                                                                        non_maxium_suppression_threshold=non_maxium_suppression_threshold)
                                                                        
        single_support_df, association_rules_df = self.frequent_object_set(object_detection_decoded_predictions)
        

        # Single Image View
        single_images_view =  self.images_properties()
        single_images_view['sentiment'] = predicted_sentiment_class
        single_images_view['style'] = predicted_style_class
        single_images_view['scence'] = predicted_scence_class
        single_images_view['indoor/outdoor'] = predicted_scene_cat_class
        single_images_view['objects'] = object_detection_decoded_predictions

        detected_objects_df = check_objects_from_df(single_images_view, basket_col='objects', objects=tracked_objs, count=False, prefix='detected_')

        single_images_view = pd.concat([single_images_view, detected_objects_df], axis=1)

        return  {
                    'Single Image View': single_images_view,            
                    'Prob Sentiment Analysis':result_sentiment_df, 
                    'Prob Style Analysis': result_style_df, 
                    'Prob Scene Analysis': result_scene_df, 
                    'Prob Scene Cat Analysis': result_scene_cat_df,
                    'Support': single_support_df, 
                    'Association Rules': association_rules_df,
                    'image_path': self.list_of_image_path
                }

if __name__ == '__main__':

    pass


    