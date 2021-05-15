import numpy as np
import cv2
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules 
import pandas as pd

class CountObjectImage():

    def __init__(self, net, labels, confident_constant = 0.5, threshold_constant = 0.3, img_width = 416, img_hight = 416):
        
        self.labels = labels
        self.net = net
        self.confident_constant = confident_constant
        self.threshold_constant = threshold_constant
        self.img_width = img_width
        self.img_hight = img_hight
        self.img = None
        self.boxes = None
        self.confidences = None
        self.classIDs = None
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype=np.uint8)
        self.olayer_name = [ self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers() ]
        self.object_count_lst = []
        self.total_obj_count = 0
    
    def fit(self, img):

        # get image
        self.img = img

        # Be careful, each model requires different preprocessing!!!
        h,w = self.img.shape[:2]
        blob = cv2.dnn.blobFromImage(self.img,
                                    1 / 255.0,                          # scaleFactor
                                    (self.img_width, self.img_hight),   # spatial size of the CNN
                                    swapRB=True, crop=False)
        # Pass the blob to the network
        self.net.setInput(blob)
        outputs = self.net.forward(self.olayer_name)

        # Lists to store detected bounding boxes, confidences and classIDs
        self.boxes = []
        self.confidences = []
        self.classIDs = []

        # Loop over each of the layer outputs
        for output in outputs:
            # Loop over each of the detections
            for detection in output:
                # Extract the confidence (i.e., probability) and classID
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
            
                # Filter out weak detections by ensuring the confidence is greater than the threshold
                if confidence < self.confident_constant:
                    continue

                # Compute the (x, y)-coordinates of the bounding box
                box = detection[0:4] * np.array( [w,h,w,h] )
                (centerX, centerY, width, height) = box.astype('int')
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Add a new bounding box to our list
                self.boxes.append([x, y, int(width), int(height)])
                self.confidences.append(float(confidence))
                self.classIDs.append(classID)

    def predict(self):

        # Count Detected Object
        labels_map = [self.labels[i] for i in self.classIDs]

        # Count each labels
        object_count = {}  
        for l in labels_map:
            if l in object_count:
                object_count[l] += 1
            else:
                object_count[l] = 1

        # Return dictionary of obj count
        return object_count

    def fit_predict(self, imgs:list ):
        
        for im in imgs:
            self.fit(im)
            obj_dic = self.predict()
            self.object_count_lst.append(obj_dic)
        return self.object_count_lst

    def summary_table(self):
        dic_binary_obj = {}
        dic_total_obj = {}
        no_input_imgs = len(self.object_count_lst)
        for obj_dic in self.object_count_lst:
            unq_obj = list(obj_dic.keys())
            for obj in unq_obj:
                if obj in dic_binary_obj:
                    dic_binary_obj[obj] += 1 
                else:
                    dic_binary_obj[obj] = 1 

            for obj2, c in obj_dic.items():
                if obj2 in dic_total_obj:
                    dic_total_obj[obj2] += c
                else:
                    dic_total_obj[obj2] = c

        df_binary_obj = pd.DataFrame({
                                        'Object' : list(dic_binary_obj.keys()),
                                        'CountImg' : list(dic_binary_obj.values())   
                                    })
        

        df_total_obj = pd.DataFrame({
                                        'Object' : list(dic_total_obj.keys()),
                                        'CountAll' : list(dic_total_obj.values())
                                     })

        df_summary = df_total_obj.merge(df_binary_obj, how = 'left', on = 'Object')
        df_summary['SupportAll'] = df_summary['CountAll'] / df_summary['CountAll'].sum()                    
        df_summary['SupportImg'] = df_summary['CountImg'] / no_input_imgs 
        df_summary.sort_values(by = 'SupportImg', ascending=False, inplace=True)
        df_summary.reset_index(drop=True, inplace=True)
        return df_summary

    def get_freq_items(self, min_support = 0.1):
        df = pd.DataFrame(self.object_count_lst)
        df_binary = df.applymap(lambda x:1 if x >= 1 else 0 )
        freq_items = frq_items = apriori(df_binary, min_support = min_support, use_colnames = True) 
        result = association_rules(freq_items, min_threshold = 1)
        result['antecedents'] = result['antecedents'].apply(lambda x:[i for i in x])
        result['consequents'] = result['consequents'].apply(lambda x:[i for i in x])
        # result['freq_object'] = result['antecedents'] + result['consequents'] 
        # result['freq_object'] = result['freq_object'].apply(lambda x:sorted(x))
        # result['freq_object'] = result['freq_object'].astype(str)
        # final_result = result.drop_duplicates(subset=['freq_object'])[['freq_object','support','confidence','lift']].sort_values(by='support', ascending=False)
        # final_result.columns = ['FreqObject', 'Support', 'Confidence', 'Lift']
        result.sort_values(by='support', ascending=False, inplace = True)
        result.reset_index(drop=True, inplace=True)
        result = result[['antecedents','consequents','support', 'confidence','lift']]
        result['support'] = result['support'].apply(lambda x:round(x,4))
        result['confidence'] = result['confidence'].apply(lambda x:round(x,4))
        result['lift'] = result['lift'].apply(lambda x:round(x,4))
        result.columns = ['Antecedents','Consequents','Support', 'Confidence','Lift']
        return result

    def image_show(self, figsize=(12,8), title = 'Image After Doing Object Detection\n', fontsize=25):

        # Non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.confident_constant, self.threshold_constant)

        # If at least one detection exists, draw the detection result(s) on the input image
        if len(idxs) > 0:
            for i in idxs.flatten():
                # Extract the bounding box coordinates
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])

                # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.colors[self.classIDs[i]]]
                cv2.rectangle(self.img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[self.classIDs[i]], self.confidences[i])
                cv2.putText(self.img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display image after detected
        plt.figure(figsize = figsize)
        plt.title(title, fontsize=fontsize)
        plt.imshow( cv2.cvtColor( self.img, cv2.COLOR_BGR2RGB) )
        plt.show()

if __name__ == "__main__":

    # ------------------------------------------------------------------------------------------
    # Initial
    # ------------------------------------------------------------------------------------------

    # Load labels of model
    labels = open("./training_model/YOLO-COCO/coco.names").read().strip().split("\n")

    # Load a pre-trained YOLOv3 model from disk
    net = cv2.dnn.readNetFromDarknet("./training_model/YOLO-COCO/yolov3.cfg","./training_model/YOLO-COCO/yolov3.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 
    # load image
    dir_img = './training_model/image_test1.jpg'
    dir_imgs = [
                    './training_model/image_test1.jpg', 
                    './training_model/image_test2.jpg', 
                    './training_model/image_test3.jpg', 
                    './training_model/image_test4.jpg', 
                    './training_model/image_test5.jpg'
                ]
    img = cv2.imread(dir_img)
    imgs = [ cv2.imread(i) for i in dir_imgs ]
    # ------------------------------------------------------------------------------------------
    # Single image process
    # ------------------------------------------------------------------------------------------
    
    co = CountObjectImage(net = net, labels = labels)
    co.fit(img)
    dic_obj = co.predict()
    print('----- Result of Single Image Process -----------------------------------------------')
    print(f' Object list : {dic_obj}')
    print('-'*84)
    # co.image_show()
    
    # ------------------------------------------------------------------------------------------
    # Batch Process
    # ------------------------------------------------------------------------------------------
    
    co = CountObjectImage(net = net, labels = labels)
    result = co.fit_predict(imgs = imgs)
    summary_table = co.summary_table()
    freq_items_set = co.get_freq_items(min_support = 0.3)
    print('-----Table Result of Batch Process--------------------------------------------------------')
    print(f' Summary Table')
    print(summary_table)
    print(f' Freqent Item Sets df')
    print(freq_items_set)
    print('-'*84)
    