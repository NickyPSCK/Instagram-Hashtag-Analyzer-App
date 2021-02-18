from datetime import datetime, timezone
from instagram_scraper import InstagramScraper
from flickr_scraper import FlickrScraper
from image_analyzer import ImageAnalyzer

import tensorflow

class HashtagAnalyzer:
    def __init__(   self, 
                    user:str, 
                    password:str,
                    
                    sentiment_classifier_path:str,
                    sentiment_classifier_pre_prep_func:object,
                    sentiment_classifier_class_label:list,

                    style_classifier_path:str,
                    style_classifier_pre_prep_func:object,
                    style_classifier_class_label:list
                    
                    ):
        
        self.__ig_scraper = InstagramScraper(user=user, password=password)
        self.__flickr_scraper = FlickrScraper()

        self.__analyzer = ImageAnalyzer(
                                            sentiment_classifier_path=sentiment_classifier_path,
                                            sentiment_classifier_pre_prep_func=sentiment_classifier_pre_prep_func,
                                            sentiment_classifier_class_label=sentiment_classifier_class_label,

                                            style_classifier_path=style_classifier_path,
                                            style_classifier_pre_prep_func=style_classifier_pre_prep_func,
                                            style_classifier_class_label=style_classifier_class_label
        )

        self.downloaded_path_ig = 'static/downloads/hashtag/instagram/'
        self.downloaded_path_flickr = 'static/downloads/hashtag/flickr/'

    def analyze_ig(self, hashtag, limit=5):
        # Download Image
        prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")

        print('Downloading..')
        self.__ig_scraper.download_from_hashtag(hashtag=hashtag, path=self.downloaded_path_ig, prefix=prefix, limit=limit)
        print('Downloaded')

        print('Analyzing..')
        result_dir = f'{self.downloaded_path_ig}{prefix}_{hashtag}/'
        target_result_dir = f'{result_dir}*.jpg'

        self.__analyzer.load_image(target_result_dir)
        result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df = self.__analyzer.analyze()
        print('Analyzed')

        return result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df

    def analyze_flickr(self, hashtag, sort_by='relevance', limit=5):
        # Download Image
        prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")

        print('Downloading..')
        self.__flickr_scraper.download_from_hashtag(hashtag, path=self.downloaded_path_flickr, sort_by=sort_by, limit=limit, per_page=100, prefix=prefix)
        print('Downloaded')

        print('Analyzing..')
        result_dir = f'{self.downloaded_path_flickr}{prefix}_{hashtag}/'
        target_result_dir = f'{result_dir}*.jpg'

        self.__analyzer.load_image(target_result_dir)
        result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df = self.__analyzer.analyze()
        print('Analyzed')

        return result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df

    def analyze_demo(self, demo_id=1):

        result_dir = f'static/downloads/hashtag/demo/demo{demo_id}/'
        target_result_dir = f'{result_dir}*.jpg'

        print('Analyzing..')
        self.__analyzer.load_image(target_result_dir)
        result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df = self.__analyzer.analyze()
        print('Analyzed')

        return result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df

if __name__ == '__main__':
    from config_loader import ConfigLoader

    cl = ConfigLoader()
    user = cl.get('login', 'user', data_type=str)
    password = cl.get('login', 'password', data_type=str)

    analyzer = HashtagAnalyzer(     
                                    user=user, 
                                    password=password,
                                    sentiment_classifier_path='model/sentiment_classification.h5',
                                    sentiment_classifier_pre_prep_func = tensorflow.keras.applications.efficientnet.preprocess_input,
                                    sentiment_classifier_class_label=['Highly_Negative', 'Negative', 'Neutral', 'Positive', 'Highly_Positive'],

                                    style_classifier_path='model/style_classification.h5',
                                    style_classifier_pre_prep_func = tensorflow.keras.applications.efficientnet.preprocess_input,
                                    style_classifier_class_label=[  'Bokeh','Bright','Depth_of_field','Detailed','Ethereal','Geometric_composition',
                                                                        'Hazy', 'Hdr', 'Horror', 'Long_exposure', 'Macro', 'Melancholy', 'Minimal', 'Noir', 
                                                                        'Pastel', 'Romantic', 'Serene', 'Sunny', 'Texture','Vintage']  
                            )


    result_sentiment_df, result_style_df, summary_frequent_object_table_df, freq_items_set_df = analyzer.analyze_flickr('apple', limit=5)
    print(result_sentiment_df)
    print(result_style_df)
    print(summary_frequent_object_table_df)
    print(freq_items_set_df)

    pass