# waifuGAN project

** IN PROGRESS **
This is a DCGAN network trained to generate anime faces. Done before, but this time using a different dataset and pytorch. 

## Data Collection

Data was scraped from [zerochan.net](https://www.zerochan.net/). There's a lot of high-quality existing datasets out there like [this](https://github.com/bchao1/Anime-Face-Dataset) and [this](https://www.kaggle.com/subinium/highresolution-anime-face-dataset-512x512), but none of them that I found used zerochan so I thought I'd try it out. I did try pixiv at first just because the art on there is cool/high quality, but I got screwed with 403 errors. Zerochan worked pretty well for this project though.
There's a bunch of pages on Zerochan: recent, popular (with different date range options), etc. Through just browsing, I decided to only use the following pages to scrape for images:
1. Popular (current day)
2. Popular (last week)
3. Popular (last 3 months)

I didn't use popular all time and recent because they were populated with both lower quality images and also very large images that were too big for my dataset. The datase I used was collected/scraped on August 10th, 2021. 

## Dataset Preparation
I used [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) as my face detector. Not perfect, but works well enough. <br>
It turns out Pytorch's resize transform is more powerful than I expected. Didn't know it could scale up, so initial implementations included checking for faces that were too small, but the current implementation doesn't use this and insteads takes all images, regardless of size. Very nice and beefs up the dataset a lot. 

## Limitations/Future Improvements
### Different art styles
![style-1](assets/images/style1.jpg "Face style 1") ![style-2](assets/images/style2.jpg "Face style 2")

Since I used an open source website and people draw art/anime in different ways, there's a lot of variety in how the faces are drawn (i.e. noise for the model). It makes it more difficult to model compared to human faces or a homogenous (maybe only from one artist) dataset. The face identifier/cropper seems to only be trained on a subset of anime faces, so it only selects what it recognizes as a face, but as displayed above, there is still a drastic difference between the art styles. 
### Implementing my own face detector
Instead using someone else's (mildly outdated) face detector, I plan on creating my own face detector. For the future...

## Extra Comments
This project was done Summer of 2021 as a fun side project to learn about DCGANs. Been having a lot of fun with AI - DCGAN theory is genius in particular. My dream is to one day use AI to change the world, instead of generating bad anime faces...
