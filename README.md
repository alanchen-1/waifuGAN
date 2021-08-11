# waifuGAN project

lol will update

## Data Collection

Data was scraped from [zerochan.net](https://www.zerochan.net/). There's a lot of high-quality existing datasets out there like [this](https://github.com/bchao1/Anime-Face-Dataset) and [this](https://www.kaggle.com/subinium/highresolution-anime-face-dataset-512x512), but none of them that I found used zerochan so I thought I'd try it out. I did try pixiv at first just because the art on there is cool/high quality, but I got screwed with 403 errors. Zerochan worked pretty well for this project though.
There's a bunch of pages on Zerochan: recent, popular (with different date range options), etc. Through just browsing, I decided to only use the following pages to scrape for images:
1. Daily popular page
2. Popular (last week)
3. Popular (last 3 months)

I didn't use popular all time and recent because they were populated with both lower quality images and also very large images that were too big for my dataset. 




Turns out transforms.Resize() is more powerful than I expected. Didn't know it could scale up, so initial implementations included checking for small images, but the current implementation doesn't use this and insteads takes all images, regardless of size. Very nice. 

![style-1](style1.png "Face style 1") ![style-2](style2.png "Face style 2")
One issue with taking the images from an open source art website is that people draw anime in different ways, so there's a lot of variety in how the faces are drawn (i.e. noise for the model). It makes it more difficult to model compared to human faces or a homogenous dataset. The face identifier/cropper seems to only be trained on a subset of anime faces, so it only selects what it recognizes as a face, but as displayed above, there is still a drastic difference between the art styles of the two displayed faces. 
