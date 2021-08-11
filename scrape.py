import os
import shutil
import requests
from bs4 import BeautifulSoup

total_count = 0
# choose directory to scrape from
# I think popular might result in higher quality images
# last week/past 3 months > all time
#recents_url = 'https://www.zerochan.net/?p='
last_week_url = 'https://www.zerochan.net/?s=fav&t=1&p='
last_months_url = 'https://www.zerochan.net/?s=fav&t=2&p='
popular_url = 'https://www.zerochan.net/popular'
page_urls = [last_week_url, last_months_url]
root_dir = './raw_images/'
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)

def download_imgs(url, directory_name = 'tmp'):
    page_count = 0
    out_dir = os.path.join(root_dir, directory_name)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    try:
        page_req = requests.get(url)
        page_soup = BeautifulSoup(page_req.text, 'html.parser')
        character_tags = [tag.attrs['src'][0:] for tag in page_soup.find_all('img')]
        if len(character_tags) > 0:
            for character in character_tags:
                r = requests.get(character, stream = True)

                if r.status_code == 200:
                    page_count += 1
                    r.raw_decode_content = True
                    filename = os.path.join(out_dir, "{}_{}.jpg".format(directory_name, str(page_count).zfill(2)))

                    with open(filename, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)

                    print('successful, ', filename)
                else:
                    print("failed, error: ", r.status_code)
                    print("moving on...")
        else:
            print("no images found @ ", url)
    except:
        print("error in getting url probably, moving on: ", url)
    return page_count


def download_imgs_page(page, page_urls):
    page_count = 0
    out_dir = os.path.join(root_dir, page)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for url in page_urls:
        try:
            page_req = requests.get(url + page)
            page_soup = BeautifulSoup(page_req.text, 'html.parser')

            character_tags = [tag.attrs['src'][0:] for tag in page_soup.find_all('img')]
            if len(character_tags) > 0:
                for character in character_tags:
                    r = requests.get(character, stream = True)

                    if r.status_code == 200:
                        page_count += 1
                        r.raw_decode_content = True
                        filename = os.path.join(out_dir, "{}_{}.jpg".format(page, str(page_count).zfill(2)))

                        with open(filename, 'wb') as f:
                            shutil.copyfileobj(r.raw, f)

                        
                        print('successful, ', filename)
                    else:
                        print("failed, error: ", r.status_code)
                        print("moving on...")
            else:
                print("no images found @ ", url)
        except:
            print("error in url probably, moving on", url)
    return page_count

initial_page = 1
final_page = 2
pages = [str(p) for p in range(initial_page, final_page+1)]
total_count += download_imgs(popular_url, directory_name="popular")
for page in pages:
    total_count += download_imgs_page(page, page_urls=page_urls)


print("Total raw scraped images: ", total_count)
