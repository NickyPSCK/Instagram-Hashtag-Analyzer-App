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
    summary_df = summary_df.reset_index(drop=True)
    return round_df(summary_df, decimals=decimals)

def summary_check_objects(check_df, decimals:int=2):
    check_df = check_df.drop('path', axis=1)
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
        
        # result_df['path'] = self.list_of_image_path

        # path = result_df['path']

        # result_df = result_df.drop(labels=['path'], axis=1)
        # result_df.insert(0, 'path', path)

        result_df.insert(0, 'path', self.list_of_image_path)

        return result_df

    def create_detected_objects_df(self, object_detection_decoded_predictions, tracked_objs:list=None, count:bool=False):
        detected_objects_df = pd.DataFrame()
        detected_objects_df['pathh'] = self.list_of_image_path
        detected_objects_df['objects'] = object_detection_decoded_predictions
        detected_objects_df = check_objects_from_df(detected_objects_df, basket_col='objects', objects=tracked_objs, count=count, prefix='')
        detected_objects_df.insert(0, 'path', self.list_of_image_path)
        return detected_objects_df

    def calculate_score(self, result_dict,    
                                        expected_sentiment:str=None,
                                        expected_style:list=None, 
                                        expected_scene:list=None, 
                                        expected_scene_cat:list=None,
                                        ):
        result_score = dict()

        # if len(expected_sentiment) == 0 :
        #     expected_sentiment = None
        # if len(expected_style) == 0 :
        #     expected_style = None
        # if len(expected_scene) == 0 :
        #     expected_scene = None
        # if len(expected_scene_cat) == 0 :
        #     expected_scene_cat = None

        n = len(result_dict['image_path'])
        decimals = 2

        if expected_sentiment is not None:
            sentiment_score = result_dict['Prob Sentiment Analysis'].iloc[:,1:].sum().to_frame(name='score')/n
            sentiment_score = sentiment_score.reset_index(drop=False)

            if expected_sentiment == 'Positive': 
                factor = {'Highly Positive': 4, 'Positive':3, 'Neutral':2, 'Negative':1, 'Highly Negative':0}
                sentiment_score['factor'] = sentiment_score['index'].map(factor)
                sentiment_score = (sentiment_score['factor'] * sentiment_score['score']).sum()/4

            elif expected_sentiment == 'Negative':
                factor = {'Highly Positive': 0, 'Positive':1, 'Neutral':2, 'Negative':3, 'Highly Negative':4}
                sentiment_score['factor'] = sentiment_score['index'].map(factor)
                sentiment_score = (sentiment_score['factor'] * sentiment_score['score']).sum()/4

            elif expected_sentiment == 'Neutral':
                factor = {'Highly Positive':2, 'Positive':1, 'Neutral':0, 'Negative':1, 'Highly Negative':2}
                sentiment_score['factor'] = sentiment_score['index'].map(factor)
                sentiment_score = 1 - (sentiment_score['factor'] * sentiment_score['score']).sum()/2        

            result_score['Sentiment'] = round(sentiment_score*100, decimals)
        
        if expected_style is not None:
            no_style = len(expected_style)
            style_score = result_dict['Prob Style Analysis'].iloc[:,1:].sum()[expected_style]/n
            style_score[style_score > 1/no_style] = 1/no_style
            style_score = style_score.sum()

            result_score['Style'] = round(style_score*100, decimals)

        if expected_scene is not None:
            no_scene = len(expected_scene)
            scene_score = result_dict['Prob Scene Analysis'].iloc[:,1:].sum()[expected_scene]/n
            scene_score[scene_score > 1/no_scene] = 1/no_scene
            scene_score = scene_score.sum()

            result_score['Scene'] = round(scene_score*100, decimals)

        if expected_scene_cat is not None:
            no_scene_cat = len(expected_scene_cat)
            scene_cat_score = result_dict['Prob Scene Cat Analysis'].iloc[:,1:].sum()[expected_scene_cat]/n
            scene_cat_score[scene_cat_score > 1/no_scene_cat] = 1/no_scene_cat
            scene_cat_score = scene_cat_score.sum()

            result_score['Outdoor/Indoor'] = round(scene_cat_score*100, decimals)

        object_score = summary_check_objects(result_dict['Detected Objects'], decimals=None).drop('count', axis=1)
        object_score['%'] = object_score['%']
        object_score = round_df(object_score, decimals=decimals)
        result_object_score = dict(object_score.to_records(index=False))
        print(result_object_score)
        result_overall = dict()
        result_overall.update(result_score)
        result_overall.update(result_object_score)
        result_overall = tuple(result_overall.values())

        if len(result_overall) > 0:
            result_overall = round((sum(result_overall)/len(result_overall)), decimals)
        else:
            result_overall = 0.0

        return {'overall': result_overall, 'expected':result_score, 'object':result_object_score}

    @__check_load_image
    def analyze(self, 
                tracked_objs:list=None, 
                expected_sentiment:str=None,
                expected_style:list=None, 
                expected_scene:list=None, 
                expected_scene_cat:list=None,
                confident_threshold:float=0.5, 
                non_maxium_suppression_threshold:float=0.3):

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
                                                                        
        detected_objects_df = self.create_detected_objects_df(object_detection_decoded_predictions, tracked_objs=tracked_objs, count=False)

        single_support_df, association_rules_df = self.frequent_object_set(object_detection_decoded_predictions)
        
        # Single Image View
        single_images_view =  self.images_properties()
        single_images_view['sentiment'] = predicted_sentiment_class
        single_images_view['style'] = predicted_style_class
        single_images_view['scence'] = predicted_scence_class
        single_images_view['indoor/outdoor'] = predicted_scene_cat_class
        single_images_view['objects'] = object_detection_decoded_predictions
        single_images_view = pd.concat([single_images_view, detected_objects_df.drop('path', axis=1)], axis=1)

        result_dict =   {
                            'Single Image View': single_images_view,            
                            'Prob Sentiment Analysis':result_sentiment_df, 
                            'Prob Style Analysis': result_style_df, 
                            'Prob Scene Analysis': result_scene_df, 
                            'Prob Scene Cat Analysis': result_scene_cat_df,
                            'Detected Objects': detected_objects_df,
                            'Support': single_support_df, 
                            'Association Rules': association_rules_df,
                            'image_path': self.list_of_image_path
                        }

        score = self.calculate_score( result_dict = result_dict, 
                            expected_sentiment=expected_sentiment,
                            expected_style=expected_style, 
                            expected_scene=expected_scene,
                            expected_scene_cat=expected_scene_cat,
                            
                            )
        return result_dict, score
if __name__ == '__main__':

    pass


    