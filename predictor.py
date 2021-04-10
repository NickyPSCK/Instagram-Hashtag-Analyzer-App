# predictor.py
# -------------------------------------------------------------------------------------------------------- 
# INDEPENDENT STUDY: HASHTAG ANALYZER
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------
import glob
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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
        elif model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self.__load_model(model_path)

        self.preprocess_input = preprocess_input
        self.__class_label = class_label

    def __class_label_tolist(self, label_dict:dict):
        label = list(label_dict.items())
        label_sorted = sorted(label, key= lambda x: int(x[0]))
        return [label[1] for label in label_sorted]

    def __load_model(self, model_path):
        return load_model(model_path)

    def load_image(self, path, color_mode='rgb'):     
        list_of_image_path = glob.glob(path)
        list_of_image_path = ['/'.join(str(Path(str_path)).split('\\')) for str_path in list_of_image_path] 

        all_input_arr = list()
        for image_path in list_of_image_path:
            image = load_img(image_path, color_mode=color_mode, target_size=(224, 224))
            input_arr = img_to_array(image)
            input_arr = np.expand_dims(input_arr, axis=0)
            all_input_arr.append(input_arr)
        input_arrs = np.vstack(all_input_arr)
        return input_arrs, list_of_image_path

    def predict(self, X):
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
        class_label = self.__class_label.copy()

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
# ObjectDetectionPrediction
# --------------------------------------------------------------------------------------------------------

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

class ObjectDetectionPrediction(ClassificationPredictor):


    def __init__(self, 
                model:object=None, 
                model_path:str=None, 
                class_label:dict=None):

        ClassificationPredictor.__init__(self, model, model_path, None, class_label)

    def __load_model(self, model_path):
        return models.load_model(model_path)

    def load_image(self, path, color_mode='bgr'): 
        return ClassificationPredictor.load_image(self, path, color_mode): 

    def predict(self, X):
        if self.model is None:
            raise Exception('Model not found.')

        images = list()
        for image in X:
            image, scale = resize_image(image)
            image = np.expand_dims(image, axis=0)
            images.append(image)

        X = np.vstack(images)
        X = preprocess_image(X)

        predictions = self.model..predict_on_batch(X)
        return predictions

    def decode_prediction(self, prediction):

        boxes, scores, labels = prediction
        exist_labels = labels[labels > -1].copy()
        exist_scores = scores[scores > -1].copy()
        exist_boxes = list(np.reshape(boxes[boxes>-1], (-1, 4)))

        result_df = pd.DataFrame(columns = ['labels_code', 'score', 'box'])
        

        result_df['labels_code'] = exist_labels
        result_df['score'] = exist_scores
        result_df['box'] = exist_boxes

        result_df.astype({'labels_code': 'str'})

        result_df['labels'] = result_df['labels_code'].map(self.__class_label)

        return result_df

    def decode_predictions(self, predictions, threshold=0.6):
        
        decoded - list()
        for prediction in predictions:
           prediction_df = self.decode_prediction(prediction)

           selected_prediction_df = prediction_df[prediction_df['score'] > threshold]['labels']
           decoded.append(tuple(set(selected_prediction_df)))
        return decoded
            
if __name__ == '__main__':

    pass




