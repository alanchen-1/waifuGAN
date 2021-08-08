import os
import shutil
import requests
from bs4 import BeautifulSoup

total_count = 0
root_url = 'https://www.zerochan.net/?p='
root_dir = './raw_images/'
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)

def download_imgs(page):
    page_count = 0
    out_dir = os.path.join(root_dir, page)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    page_req = requests.get(root_url + page)
    page_soup = BeautifulSoup(page_req.text, 'html.parser')

    character_tags = [tag.attrs['src'][0:] for tag in page_soup.find_all('img')]
    for character in character_tags:
        r = requests.get(character, stream = True)

        if r.status_code == 200:
            r.raw_decode_content = True
            filename = os.path.join(out_dir, "{}_{}.jpg".format(page, str(page_count).zfill(2)))

            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

            page_count += 1
            print('successful, ', filename)
        else:
            print("failed, error:", r.status_code)
            print("moving on...")
    return page_count

initial_page = 1
final_page = 1000
pages = [str(p) for p in range(initial_page, final_page+1)]
for page in pages:
    download_imgs(page)
