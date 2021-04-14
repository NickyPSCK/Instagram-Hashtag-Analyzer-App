# instragram_scraper.py
# -------------------------------------------------------------------------------------------------------- 
# INDEPENDENT STUDY: HASHTAG ANALYZER
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------
import os
import instaloader
from datetime import datetime, timezone

# --------------------------------------------------------------------------------------------------------
# InstagramScraper
# --------------------------------------------------------------------------------------------------------
class InstagramScraper:
    def __init__(self, user, password):
        self.__il = instaloader.Instaloader(
                                        sleep=True, 
                                        quiet=False, 
                                        user_agent=None, 
                                        dirname_pattern=None, 
                                        filename_pattern=None, 
                                        download_pictures=True, 
                                        download_videos=False, 
                                        download_video_thumbnails=False, 
                                        download_geotags=True, 
                                        download_comments=True, 
                                        save_metadata=True, 
                                        compress_json=True, 
                                        post_metadata_txt_pattern=None, 
                                        storyitem_metadata_txt_pattern=None, 
                                        max_connection_attempts=3, 
                                        request_timeout=None, 
                                        rate_controller=None, 
                                        resume_prefix='iterator', 
                                        check_resume_bbd=True)

        self.__il.login(user, password) 

    def download_from_hashtag(self, hashtag, path='downloads/hashtag/instagram/', prefix=None, limit=10):

        if prefix is None:
            prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        hashtag_data = instaloader.Hashtag.from_name(self.__il.context, hashtag)
        self.__il.dirname_pattern = f'{path}{prefix}_{hashtag_data.name}/'
        no = 0
        for post in hashtag_data.get_posts():

            if no > limit:
                break

            self.__il.download_post(post, target='/')
            no += 1


if __name__ == '__main__':

    pass

