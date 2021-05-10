# app.py
# -------------------------------------------------------------------------------------------------------- 
# INDEPENDENT STUDY: HASHTAG ANALYZER
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow
from flask import Flask, request, render_template

from util.init_sys import InitSystem
from util.utility import round_df
# from util.config_loader import ConfigLoader

from hashtag_analyzer import HashtagAnalyzer

app = Flask(__name__)

# -------------------------------------------------------------------------------------------
# Helpper Function
# -------------------------------------------------------------------------------------------

def process_result(**dfs):
    df_list = list()
    for df in dfs:
        table_head = list(dfs[df].columns)
        table_data = dfs[df].to_records(index=False)

        name = ' '.join(df.split('_'))
        df_list.append({'name':name, 'head':table_head, 'data':table_data})
    return df_list

def convert_pattern(data_wo_path_list):
    # -------------------------------------------------------------------------------------------
    # Process result_sentiment_df and result_style_df
    # -------------------------------------------------------------------------------------------
    data_wo_path_list_of_tupple = list()
    for prediction in data_wo_path_list:

        prediction_tuple = [(k, round(v,4)) for k, v in prediction.items()]
        sorted_result = sorted(prediction_tuple, key=lambda i: i[1], reverse=True) 
        data_wo_path_list_of_tupple.append(sorted_result)

    return data_wo_path_list_of_tupple

def convert_returned_args(args):

    args_dict = args.to_dict(flat=False)
    for arg in args_dict:
        if len(args_dict[arg]) == 1:
            args_dict[arg] = args_dict[arg][0]

    return args_dict

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['GET'])
def result():

    # form_data = request.args.to_dict()

    form_data = convert_returned_args(request.args)

    hashtag = form_data["hashtag"]
    source = form_data["source"]
    limit = int(form_data["limit"])

    if source == 'Flickr':
        analysis_result = analyzer.analyze_flickr(hashtag=hashtag, limit=limit)
    elif source == 'Instagram':
        analysis_result = analyzer.analyze_ig(hashtag=hashtag, limit=limit)
    elif source == 'Demo1':
        hashtag = 'DEMO1'
        analysis_result = analyzer.analyze_demo(demo_id=1)
    elif source == 'Demo2':
        hashtag = 'DEMO2'
        analysis_result = analyzer.analyze_demo(demo_id=2)
    elif source == 'Demo3':
        hashtag = 'DEMO3'
        analysis_result = analyzer.analyze_demo(demo_id=3)
    else:
        analysis_result = analyzer.analyze_demo(demo_id=1)

    summary_result_tables = dict()
    raw_result_tables = list()

    for table_name in analysis_result:
        if 'image_path' not in table_name.lower():
            if 'raw' not in table_name.lower():
                summary_result_tables[table_name] = round_df(df=analysis_result[table_name], decimals=2)

            else:
                raw_df = analysis_result[table_name].drop('path', axis=1)
                raw_list = raw_df.to_dict('records')
                raw_list_of_tupple = convert_pattern(raw_list)
                raw_result_tables.append(raw_list_of_tupple)

    img_path_list = analysis_result['image_path']

    processed_img_path_list = list()
    for img_path in img_path_list:
        img_path = '/'.join(img_path.split('/')[1:])
        processed_img_path_list.append(img_path)

    result_tables = process_result(**summary_result_tables)
    detail_result_table = list(zip(processed_img_path_list, *raw_result_tables))

    # -------------------------------------------------------------------------------------------

    return render_template('result.html',
                            hashtag = hashtag,
                            source  = source,
                            result_tables = result_tables,
                            detail_result_table = detail_result_table,
                            show_result = True)

if __name__ == "__main__":

    import json

    cl = InitSystem().init()

    user = cl.get('login', 'user', data_type=str)
    password = cl.get('login', 'password', data_type=str)

    with open('config/class.json', 'r') as f:
        class_label = json.loads(f.read())

    analyzer = HashtagAnalyzer(     
                                    user = user, 
                                    password = password,
                                    sentiment_classifier_path='model/sentiment_classification.h5',
                                    sentiment_classifier_pre_prep_func = tensorflow.keras.applications.efficientnet.preprocess_input,
                                    sentiment_classifier_class_label = class_label['sentiment_classification_label'],

                                    style_classifier_path = 'model/style_classification.h5',
                                    style_classifier_pre_prep_func = tensorflow.keras.applications.efficientnet.preprocess_input,
                                    style_classifier_class_label = class_label['style_classification_label'],

                                    scene_classifier_path = 'model/scene_classification.h5',
                                    scene_classifier_pre_prep_func = None,
                                    scene_classifier_class_label = class_label['scene_classification_label'],  
                                    scene_classifier_cat_label = class_label['scene_classification_cat_label'],      

                                    object_detection_model_path = 'model/YOLO-COCO/yolov4.cfg',
                                    object_detection_model_weight_path = 'model/YOLO-COCO/yolov4.weights',
                                    object_detection_class_label = class_label['object_detection_label'],

                                    object_detection_algorithm = 'yolo',
                                    frequent_itemsets_algorithm = 'apriori',
                                    min_support = 0.3,
                                    association_metric = 'confidence',
                                    association_min_threshold = 1


                                )

    
    app.run(debug = True, port = 5000)