import os
import shutil
import requests
from bs4 import BeautifulSoup

total_count = 0
# choose directory to scrape from
# I think popular might result in higher quality images
# 15 images per page for shuu shuu
# top has 
top_url = 'https://e-shuushuu.net/top.php?page='
normal_url = 'https://e-shuushuu.net/?page='
page_urls = [top_url, normal_url]
name = 'shuushuu'
root_dir = './raw_images/'
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)

root_dir = os.path.join(root_dir, name)
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)

def download_imgs_page(page, page_urls, root_dir):
    """
    Downloads the images into the root_dir. 
        Paramters:
            page (str) : page name
            page_urls (list[str]) : page urls to iterate through
            root_dir (str) : root directory to download into
        Returns:
            page_count (int) : number of images on this page
    """
    page_count = 0
    out_dir = os.path.join(root_dir, page)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    for url in page_urls:
        try:
            page_req = requests.get(url + page)
            page_soup = BeautifulSoup(page_req.text, 'html.parser')

            img_tags = [tag.attrs['href'][0:] for tag in page_soup.find_all('a', {"class" : "thumb_image"})]
            if len(img_tags) > 0:
                for img in img_tags:
                    img = 'https://e-shuushuu.net/' + img
                    r = requests.get(img, stream = True)

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

# scrape top first
initial_page = 1
final_page = 20
pages = [str(p) for p in range(initial_page, final_page+1)]
top_dir = os.path.join(root_dir, 'top')
if not os.path.isdir(top_dir):
        os.mkdir(top_dir)
for page in pages:
    total_count += download_imgs_page(page, page_urls = [top_url], root_dir = top_dir)

# then scrape normal
initial_page = 1
final_page = 2000
pages = [str(p) for p in range(initial_page, final_page+1)]
norm_dir = os.path.join(root_dir, 'normal')
if not os.path.isdir(norm_dir):
        os.mkdir(norm_dir)

for page in pages:
    total_count += download_imgs_page(page, page_urls = [normal_url], root_dir = norm_dir)

print("Total raw scraped images: ", total_count)
