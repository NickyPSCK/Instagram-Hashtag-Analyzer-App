import glob
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import efficientnet.keras as efn

class ClassificationPredictor:

    def __init__(self, 
                model:object=None, 
                model_path:str=None, 
                preprocess_input=None,  
                class_label:list=None):

        # Initial Class
        if model is not None and model_path is not None:
            raise Exception('Please spacify either model or model_path')
        elif model is not None:
            self.model = model
        elif model_path is not None:
            self.model = load_model(model_path)

        self.preprocess_input = preprocess_input
        self.__class_label = class_label

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

        if self.__class_label is None:
            self.__class_label = range(len(predictions[0]))

        for prediction in predictions:

            result = list((zip(self.__class_label, list(prediction))))
            if top is not None:
                result = sorted(result, key=lambda i: i[1], reverse=True)[:top]
            decoded.append(result)

        return decoded
        
if __name__ == '__main__':

    # # Example
    # from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input, decode_predictions
    # predictor = ClassificationPredictor( 
    #                                             model_path='model/sentiment_classification.h5', 
    #                                             preprocess_input=preprocess_input,
    #                                             class_label=['Heightly Negative', 'Negative', 'Neutral', 'Positive', 'Heightly Positive']
    #                                             )
    # predictions, list_of_image_path = predictor.predict_form_path('test_img/*.jpg')

    # print(predictor.decode_predictions(predictions, top=3))



    from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input, decode_predictions
    predictor = ClassificationPredictor( 
                                                model_path='model/style_classification.h5', 
                                                preprocess_input=preprocess_input,
                                                class_label=[  'Bokeh','Bright','Depth_of_field','Detailed','Ethereal','Geometric_composition',
                                                            'Hazy', 'Hdr', 'Horror', 'Long_exposure', 'Macro', 'Melancholy', 'Minimal', 'Noir', 
                                                            'Pastel', 'Romantic', 'Serene', 'Sunny', 'Texture','Vintage']   
                                                )
    predictions, list_of_image_path = predictor.predict_form_path('test_img/*.jpg')

    print(predictor.decode_predictions(predictions, top=3))

    pass