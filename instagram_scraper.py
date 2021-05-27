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
    def __init__(self, user:str=None, password:str=None, login:bool=False):
        self.__il = instaloader.Instaloader(
                                        sleep=True, 
                                        quiet=False, 
                                        user_agent=None, 
                                        dirname_pattern=None, 
                                        filename_pattern=None, 
                                        download_pictures=True, 
                                        download_videos=False, 
                                        download_video_thumbnails=False, 
                                        download_geotags=False, 
                                        download_comments=False, 
                                        save_metadata=False, 
                                        compress_json=False, 
                                        post_metadata_txt_pattern=None, 
                                        storyitem_metadata_txt_pattern=None, 
                                        max_connection_attempts=3, 
                                        request_timeout=None, 
                                        rate_controller=None, 
                                        resume_prefix='iterator', 
                                        check_resume_bbd=True)

        if login:
            self.__il.login(user, password) 

    def download_from_hashtag(self, target, mode='account', path='downloads/hashtag/instagram/', prefix=None, limit=10):

        if prefix is None:
            prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        if mode == 'account':
            posts = instaloader.Profile.from_username(self.__il.context, target)
            self.__il.dirname_pattern = f'{path}{prefix}_{posts.username}/'

        elif mode == 'hashtag':
            posts = instaloader.Hashtag.from_name(self.__il.context, target)
            self.__il.dirname_pattern = f'{path}{prefix}_{posts.name}/'

        counter = 0
        for post in posts.get_posts():
            if counter > limit:
                break
            self.__il.download_post(post, target='/')
            counter += 1


if __name__ == '__main__':

    pass

