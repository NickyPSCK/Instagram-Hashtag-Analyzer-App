# app.py
# -------------------------------------------------------------------------------------------------------- 
# INDEPENDENT STUDY: 
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import tensorflow
from flask import Flask, request, render_template


from util.init_sys import InitSystem
# from util.config_loader import ConfigLoader

from hashtag_analyzer import HashtagAnalyzer


app = Flask(__name__)

def process_result(**dfs):
    df_list = list()
    for df in dfs:
        table_head = list(dfs[df].columns)
        table_data = dfs[df].to_records(index=False)
        df_list.append({'name':df, 'head':table_head, 'data':table_data})
    return df_list

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['GET'])
def result():

    form_data = request.args.to_dict()
    hashtag = form_data["hashtag"]
    source = form_data["source"]
    limit = int(form_data["limit"])

    if source == 'Flickr':
        result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df = analyzer.analyze_flickr(hashtag=hashtag, limit=limit)
    elif source == 'Instagram':
        result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df = analyzer.analyze_ig(hashtag=hashtag, limit=limit)
    elif source == 'Demo1':
        hashtag = 'DEMO1'
        result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df = analyzer.analyze_demo(demo_id=1)
    elif source == 'Demo2':
        hashtag = 'DEMO2'
        result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df = analyzer.analyze_demo(demo_id=2)
    elif source == 'Demo3':
        hashtag = 'DEMO3'
        result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df = analyzer.analyze_demo(demo_id=3)
    else:
        result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df = analyzer.analyze_demo(demo_id=1)

    df_sent_idmax = result_sentiment_df[['Highly_Negative', 'Negative', 'Neutral', 'Positive','Highly_Positive']].idxmax(axis="columns")
    df_sent_idmax = df_sent_idmax.value_counts().to_frame()
    df_sent_idmax.columns = ['CountMaxScore']
    df_sent_idmax['CountMaxRatio'] = df_sent_idmax['CountMaxScore']/len(result_sentiment_df)
    df_sent_idmax.sort_values(by='CountMaxScore', ascending=False, inplace = True)
    df_sent_idmax = df_sent_idmax.reset_index()
    df_sent_idmax.columns = ['Sentiment', 'Number of Image', 'Percentage of Image']
    df_sent_idmax['Percentage of Image'] = df_sent_idmax['Percentage of Image'].apply(lambda x: round(float(x), 4))

    df_style_idmax = result_style_df.iloc[:,1:].idxmax(axis="columns")
    df_style_idmax = df_style_idmax.value_counts().to_frame()
    df_style_idmax.columns = ['CountMaxScore']
    df_style_idmax['CountMaxRatio'] = df_style_idmax['CountMaxScore']/len(result_style_df)
    df_style_idmax.sort_values(by='CountMaxScore', ascending=False, inplace = True)
    df_style_idmax['CountMaxScore'] = df_style_idmax['CountMaxScore'].astype(int)
    df_style_idmax = df_style_idmax.reset_index()
    df_style_idmax.columns = ['Style', 'Number of Image', 'Percentage of Image']
    df_style_idmax['Percentage of Image'] = df_style_idmax['Percentage of Image'].apply(lambda x: round(float(x), 4))

    summary_frequent_object_table_df.columns = ['Object', 'Number of Object', 'Number of Image', 'Support: Object', 'Support: Image']
    summary_frequent_object_table_df['Support: Object'] = summary_frequent_object_table_df['Support: Object'].apply(lambda x: round(float(x), 4))
    summary_frequent_object_table_df['Support: Image'] = summary_frequent_object_table_df['Support: Image'].apply(lambda x: round(float(x), 4))


    freq_items_set_df.columns = ['Antecedents',	'Consequents', 'Support', 'Confidence',	'Lift']
    
    result_tables = process_result( 
                                    Sentiment=df_sent_idmax,
                                    Style=df_style_idmax,
                                    Apiori=summary_frequent_object_table_df,
                                    FrequentItems=freq_items_set_df)

    # -------------------------------------------------------------------------------------------
    # Process result_sentiment_df and result_style_df
    # -------------------------------------------------------------------------------------------
    def convert_pattern(data_wo_path_list):
        data_wo_path_list_of_tupple = list()
        for prediction in data_wo_path_list:

            prediction_tuple = [(k, round(v,4)) for k, v in prediction.items()]
            sorted_result = sorted(prediction_tuple, key=lambda i: i[1], reverse=True) 
            data_wo_path_list_of_tupple.append(sorted_result)

        return data_wo_path_list_of_tupple

    data_style_wo_path_df = result_style_df.drop('path', axis=1)
    data_style_wo_path_list = data_style_wo_path_df.to_dict('records')
    data_style_wo_path_list_of_tupple = convert_pattern(data_style_wo_path_list)

    data_sentiment_wo_path_df = result_sentiment_df.drop('path', axis=1)
    data_sentiment_wo_path_list = data_sentiment_wo_path_df.to_dict('records')
    data_sentiment_wo_path_list_of_tupple = convert_pattern(data_sentiment_wo_path_list)

    processed_img_path_list = list()
    img_path_list = result_style_df['path'].tolist()
    for img_path in img_path_list:
        img_path = '/'.join(img_path.split('/')[1:])
        processed_img_path_list.append(img_path)
    
    detail_result_table = list(zip(processed_img_path_list, data_style_wo_path_list_of_tupple, data_sentiment_wo_path_list_of_tupple))

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
                                    user=user, 
                                    password=password,
                                    sentiment_classifier_path='model/sentiment_classification.h5',
                                    sentiment_classifier_pre_prep_func = tensorflow.keras.applications.efficientnet.preprocess_input,
                                    sentiment_classifier_class_label=class_label['sentiment_classification_label'],

                                    style_classifier_path='model/style_classification.h5',
                                    style_classifier_pre_prep_func = tensorflow.keras.applications.efficientnet.preprocess_input,
                                    style_classifier_class_label=class_label['style_classification_label']
                                )

    
    app.run(debug = True, port = 5000)