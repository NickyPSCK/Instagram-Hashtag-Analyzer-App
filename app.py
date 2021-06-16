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

from hashtag_analyzer import HashtagAnalyzer
from image_analyzer import summary_single_view, summary_check_objects

app = Flask(__name__)

# -------------------------------------------------------------------------------------------
# Helpper Function
# -------------------------------------------------------------------------------------------
def merge_prob_result(path_column_name:str='path',sep:str='\n', limit:int=5, floating_point:int=4,**dfs, ):

    def merge_prob(row,path_column_name:str='path', sep:str='\n', limit:int=5, floating_point:int=4):
        row_dict = row.to_dict()
        del row_dict[path_column_name]
        row_list = list(row_dict.items())
        row_list.sort(reverse=True, key=lambda x: x[1])
        result_str = ''
        if limit > -1:
            limit+=1
        for each in row_list[:limit]:
            result_str += each[0] + ':' +str(round(each[1], floating_point)) + sep
        return result_str

    result_df = pd.DataFrame()
    for df in dfs:
        result_df[path_column_name] = dfs[df][path_column_name]
        result_df[df] = dfs[df].apply(lambda prob_df: merge_prob(prob_df, path_column_name=path_column_name, sep=sep, limit=limit), axis=1)
    return result_df

def process_result(**dfs):
    df_list = list()
    for df in dfs:
        table_head = list(dfs[df].columns)
        table_data = dfs[df].to_records(index=False)
        name = ' '.join(df.split('_'))
        df_list.append({'name':name, 'head':table_head, 'data':table_data})
    return df_list

# -------------------------------------------------------------------------------------------

@app.route('/')
def home():
    theme = 's008'
    return render_template('home.html', theme=theme, 
                                        obj_dict=class_label['object_detection_label'],
                                        style_dict=class_label['style_classification_label'],
                                        scene_dict=class_label['scene_classification_label'],  
                                        scene_cat_dict=class_label['scene_classification_cat_label'])

@app.route('/result', methods=['GET'])
def result():

    form_data = request.args.to_dict(flat=False)

    hashtag = form_data['hashtag'][0]
    source = form_data['source'][0]
    limit = int(form_data['limit'][0])
    tracked_objs = form_data.get('objs', list())
    confident_threshold = float(form_data['confident_threshold'][0])
    non_maxium_suppression_threshold = float(form_data['non_maxium_suppression_threshold'][0])
    top_n_prob = int(form_data['top_n_prob'][0])

    expected_sentiment = form_data.get('sentiments',None)[0]
    expected_style = form_data.get('styles',None)
    expected_scene = form_data.get('scenes',None)
    expected_scene_cat = form_data.get('scene_cats',None)

    if source == 'Flickr':
        source_text = 'Flickr Hashtag'
        analysis_result, score = analyzer.analyze_flickr(hashtag=hashtag, limit=limit, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)
    elif source == 'Instagram_Account':
        source_text = 'Instagram Account'
        analysis_result, score  = analyzer.analyze_ig(target=hashtag, mode='account', limit=limit, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)
    elif source == 'Instagram_Hashtag':
        source_text = 'Instagram Hashtag'
        analysis_result, score  = analyzer.analyze_ig(target=hashtag, mode='hashtag',limit=limit, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)
    elif source == 'Demo1':
        source_text = 'Demo1'
        hashtag = 'Demo1'
        analysis_result, score  = analyzer.analyze_demo(demo_id=1, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)
    elif source == 'Demo2':
        source_text = 'Demo2'
        hashtag = 'Demo2'
        analysis_result, score  = analyzer.analyze_demo(demo_id=2, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold=non_maxium_suppression_threshold)
    elif source == 'Demo3':
        source_text = 'Demo3'
        hashtag = 'Demo3'
        analysis_result, score  = analyzer.analyze_demo(demo_id=3, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)

    elif source == 'USER_errorbxx':
        source_text = 'Instagram Account'
        hashtag = 'errorbxx'
        analysis_result, score  = analyzer.analyze_demo(demo_id=4, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)
    elif source == 'USER_joeybangkokboy':
        source_text = 'Instagram Account'
        hashtag = 'joeybangkokboy'
        analysis_result, score  = analyzer.analyze_demo(demo_id=5, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)

    elif source == 'USER_apitsada':
        source_text = 'Instagram Account'
        hashtag = 'apitsada'
        analysis_result, score  = analyzer.analyze_demo(demo_id=6, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)

    elif source == 'USER_sirinissirin':
        source_text = 'Instagram Account'
        hashtag = 'sirinissirin'
        analysis_result, score  = analyzer.analyze_demo(demo_id=7, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)

    elif source == 'USER_weir_____things':
        source_text = 'Instagram Account'
        hashtag = 'weir_____things'
        analysis_result, score  = analyzer.analyze_demo(demo_id=8, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)

    elif source == 'USER_paloyh':
        source_text = 'Instagram Account'
        hashtag = 'paloyh'
        analysis_result, score  = analyzer.analyze_demo(demo_id=9, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)

    elif source == 'USER_toeyjarinporn':
        source_text = 'Instagram Account'
        hashtag = 'toeyjarinporn'
        analysis_result, score  = analyzer.analyze_demo(demo_id=10, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold=confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)

    else:
        source_text = "Demo1"
        hashtag = 'Demo1'
        analysis_result, score  = analyzer.analyze_demo(demo_id=1, tracked_objs=tracked_objs, 
                                                    expected_sentiment=expected_sentiment,
                                                    expected_style=expected_style, 
                                                    expected_scene=expected_scene,
                                                    expected_scene_cat=expected_scene_cat,
                                                    confident_threshold = confident_threshold, 
                                                    non_maxium_suppression_threshold = non_maxium_suppression_threshold)

    decimals = 2

    # -------------------------------------------------------------------------------------------
    # RESULT PART
    # -------------------------------------------------------------------------------------------

    no_of_image = len(analysis_result['Single Image View'])
    # SECTION 1
    section_1_tables = dict()
    section_1_tables['Images Size'] = summary_single_view(analysis_result['Single Image View'], key=['width', 'height'], decimals=decimals)
    section_1_tables['Aspect Ratio'] = summary_single_view(analysis_result['Single Image View'], key=['aspect ratio'], decimals=decimals)
    section_1_tables['Orientation'] = summary_single_view(analysis_result['Single Image View'], key=['orientation'], decimals=decimals)
    section_1_tables = process_result(**section_1_tables)

    # SECTION 2
    section_2_tables = dict()
    section_2_tables['Sentiment Analysis'] = summary_single_view(analysis_result['Single Image View'], key=['sentiment'], decimals=decimals)
    section_2_tables['Style Analysis'] = summary_single_view(analysis_result['Single Image View'], key=['style'], decimals=decimals)
    section_2_tables['Scene Analysis'] = summary_single_view(analysis_result['Single Image View'], key=['scence'], decimals=decimals)
    section_2_tables['Indoor/Outdoor Analysis'] = summary_single_view(analysis_result['Single Image View'], key=['indoor/outdoor'], decimals=decimals)
    section_2_tables = process_result(**section_2_tables)

    # SECTION 3
    section_3_tables = dict()
    section_3_tables['Detected Object'] = summary_check_objects(analysis_result['Detected Objects'], decimals=decimals)
    section_3_tables = process_result(**section_3_tables)
    
    # SECTION 4
    section_4_tables = dict()
    section_4_tables['Support'] = round_df(df=analysis_result['Support'], decimals=decimals)
    section_4_tables['Association Rules'] = round_df(df=analysis_result['Association Rules'], decimals=decimals)
    section_4_tables = process_result(**section_4_tables)

    # SECTION 5
    section_5_tables = dict()
    section_5_tables['Details'] = analysis_result['Single Image View']
    section_5_tables = process_result(**section_5_tables)

    # SECTION 6  
    prob_tables = {
    'Prob Sentiment Analysis': analysis_result['Prob Sentiment Analysis'],
    'Prob Style Analysis': analysis_result['Prob Style Analysis'],
    'Prob Scene Analysis': analysis_result['Prob Scene Analysis'],
    'Prob Scene Cat Analysis': analysis_result['Prob Scene Cat Analysis']
    }
    section_6_tables = dict()
    section_6_tables['Probability Details'] = merge_prob_result(path_column_name='path', sep='\n', limit=top_n_prob, floating_point=4,**prob_tables)
    section_6_tables = process_result(**section_6_tables)
    # -------------------------------------------------------------------------------------------

    return render_template('result.html',
                            hashtag = hashtag,
                            source_text  = source_text,
                            no_of_image = no_of_image,
                            score = score,
                            section_1_tables = section_1_tables,
                            section_2_tables = section_2_tables,
                            section_3_tables = section_3_tables,
                            section_4_tables = section_4_tables,
                            section_5_tables = section_5_tables,
                            section_6_tables = section_6_tables,
                            show_result = True)

if __name__ == "__main__":

    import json

    cl = InitSystem().init()

    user = cl.get('login', 'user', data_type=str)
    password = cl.get('login', 'password', data_type=str)
    theme = cl.get('web', 'theme', data_type=str)

    with open('config/class.json', 'r') as f:
        class_label = json.loads(f.read())

    analyzer = HashtagAnalyzer(     
                                    user = user, 
                                    password = password,
                                    login=False,
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

    # Ref Html
    # https://uicookies.com/bootstrap-search-box/