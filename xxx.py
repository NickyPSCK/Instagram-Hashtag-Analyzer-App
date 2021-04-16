from util.init_sys import InitSystem
InitSystem().init()
import tensorflow
from image_analyzer import ImageAnalyzer
import json
with open('config/class.json', 'r') as f:
    class_label = json.loads(f.read())


ia = ImageAnalyzer(

                    sentiment_classifier_path = 'model/sentiment_classification.h5',
                    sentiment_classifier_pre_prep_func = tensorflow.keras.applications.efficientnet.preprocess_input,
                    sentiment_classifier_class_label = class_label['sentiment_classification_label'],

                    style_classifier_path = 'model/style_classification.h5',
                    style_classifier_pre_prep_func = tensorflow.keras.applications.efficientnet.preprocess_input,
                    style_classifier_class_label = class_label['style_classification_label'],                
                    
                    scence_classifier_path = 'model/scence_classification.h5',
                    scence_classifier_pre_prep_func = None,
                    scence_classifier_class_label = class_label['scence_classification_label'],  

                    object_detection_model_path = 'model/YOLO-COCO/yolov4.cfg',
                    object_detection_model_weight_path = 'model/YOLO-COCO/yolov4.weights',
                    object_detection_class_label  =class_label['object_detection_label'],
                    object_detection_algorithm = 'yolo',

                    frequent_itemsets_algorithm = 'apriori',
                    min_support = 0.3,
                    association_metric = 'confidence',
                    association_min_threshold = 1
                    )

ia.load_image(img_glob_pathname = 'static/downloads/hashtag/demo/demo1/*.jpg')

result = ia.analyze()

print('ddd')
print(result)
