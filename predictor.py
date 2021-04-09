import glob
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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
            self.model = load_model(model_path)

        self.preprocess_input = preprocess_input
        self.__class_label = class_label

    def __class_label_tolist(self, label_dict:dict):
        label = list(label_dict.items())
        label_sorted = sorted(label, key= lambda x: int(x[0]))
        return [label[1] for label in label_sorted]

    def load_image(self, path):     
        list_of_image_path = glob.glob(path)
        list_of_image_path = ['/'.join(str(Path(str_path)).split('\\')) for str_path in list_of_image_path] 

        all_input_arr = list()
        for image_path in list_of_image_path:
            image = load_img(image_path, target_size=(224, 224))
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
    
class ObjectDetectionPrediction(ClassificationPredictor):

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
            
if __name__ == '__main__':

    pass




