# flickr_scraper.py
# -------------------------------------------------------------------------------------------------------- 
# HASHTAG ANALYZER
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------
import os
import re
import json
import requests
from datetime import datetime, timezone

# --------------------------------------------------------------------------------------------------------
# FlickrScraper
# https://www.flickr.com/services/api/
# --------------------------------------------------------------------------------------------------------
class FlickrScraper:
    def __init__(self):

        self.__sort_by_mode = ['date-posted-desc', 'relevance', 'date-taken-desc', 'interestingness-desc']
        self.__all_url_size = ['url_o', 'url_l', 'url_c', 'url_z', 'url_n', 'url_m', 'url_q', 'url_s', 'url_t', 'url_sq']
        self.__s = requests.Session()
        self.__get_api_key()

    def renew_session(self):
        self.__s = requests.Session()
        self.__get_api_key()

    def __get_api_key(self):

        url=f'https://www.flickr.com/search/'
        r = self.__s.get(url)
        if r.status_code == 200:
            print('Sucess')
        else:
            print('Fail')

        self.__site_key = re.findall(r'root.YUI_config.flickr.api.site_key = "(.*?)";',r.text)[0]
        self.__reqId = re.findall(r'root.YUI_config.flickr.request.id = "(.*?)";',r.text)[0]

    def __gen_api_link(self, tag, page=1, sort_by='relevance', per_page=100):

        if sort_by not in self.__sort_by_mode:
            raise Exception('Invalid sort_by')

        url = f'https://api.flickr.com/services/rest\
?sort={sort_by}\
&parse_tags=1\
&content_type=7\
&extras=can_comment%2Ccan_print%2Ccount_comments%2Ccount_faves%2Cdescription%2Cisfavorite%2Clicense%2Cmedia%2Cneeds_interstitial%2Cowner_name%2Cpath_alias%2Crealname%2Crotation%2Curl_sq%2Curl_q%2Curl_t%2Curl_s%2Curl_n%2Curl_w%2Curl_m%2Curl_z%2Curl_c%2Curl_l\
&per_page={per_page}\
&page={page}\
&lang=en-US\
&text={tag}\
&view_all=1\
&viewerNSID=\
&method=flickr.photos.search\
&csrf=\
&api_key={self.__site_key}\
&format=json\
&hermes=1\
&hermesClient=1\
&reqId={self.__reqId}\
&nojsoncallback=1'

        return url

    def __extract_url(self, raw_data):
        content = raw_data['photos']['photo']
        result = list()
        
        if len(content) > 0:

            for each_image in content:
                
                for url_size in self.__all_url_size:
                    if url_size in each_image.keys():
                        url = each_image[url_size]
                        break
                    else:
                        url=''

                image_id = each_image['id']
                result.append((image_id, url))

        return result

    def get_raw_url(self, tag, page=1, sort_by='relevance', per_page=100):
        url = self.__gen_api_link(tag, page=page, sort_by=sort_by, per_page=per_page)
        r = self.__s.get(url)
        return r.json()

    def get_url(self, tag, sort_by='relevance', limit=100, per_page=100):

        first_page = self.get_raw_url(tag, page=1, sort_by=sort_by, per_page=per_page)
        total_page = first_page['photos']['pages']
        total_image = first_page['photos']['pages']

        result = list()

        if limit is not None:

            if limit/per_page <= 1:
                result = self.__extract_url(first_page)
            else:
                if limit % per_page == 0:
                    last_page = limit//per_page
                else:
                    last_page = (limit//per_page)+1

                if last_page > total_page:
                    last_page = total_page 

                for page in range(2, last_page+1):
                    result += self.__extract_url(self.get_raw_url(tag, page=page, sort_by=sort_by, per_page=per_page))

            result = result[:limit]

        else:
            for page in range(2, total_page+1):
                result += self.__extract_url(self.get_raw_url(tag, page=page, sort_by=sort_by, per_page=per_page))

        return result

    def download_from_hashtag(self, tag, path='downloads/hashtag/flickr/', sort_by='relevance', limit=100, per_page=100, prefix=None):

        urls = self.get_url(tag=tag, sort_by=sort_by, limit=limit, per_page=per_page)

        if prefix is None:
            prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        path=f'{path}{prefix}_{tag}/'
        
        full_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), *path.split('/'))
        if not os.path.exists(full_path):
            os.mkdir(full_path)
            
        for img_id in urls:

            response = requests.get(img_id[1])
            with open(f'{path}{img_id[1].split("/")[-1]}', 'wb') as f:

                f.write(response.content)

if __name__ == '__main__':

    fs = FlickrScraper()
    prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fs.download_from_hashtag('apple', path='downloads/hashtag/flickr/', sort_by='relevance', limit=10, per_page=100, prefix=prefix)

    pass

