import cv2
import sys
import os.path

"""
string, string, string -> boolean
detects faces, crops them, and outputs the into files

"""
def detect(filename, page, out_dir, cascade_file = './lbpcascade_animeface.xml'):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found " % cascade_file)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    # apply cascade as demonstrated in example
    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    
    if len(faces) > 0:
        for face in range(len(faces)):
            cur_image = filename[-6:-4]
            x, y, w, h = faces[face]
            # crop face
            out_file = os.path.join(out_dir, '{}_{}_{}.jpg'.format(page, cur_image, str(face+1).zfill(2)))
            cv2.imwrite(out_file, image[int(y-0.1*h): int(y+0.9*h), x: x+w])
        return True, len(faces), out_file
    else:
        return False, 0, ''

cropped_dir = 'cropped_images'
raw_dir = './raw_images/'
if not os.path.isdir(cropped_dir):
    os.mkdir(cropped_dir)

cropped_dir = './cropped_images/'

#initial_page = 1
#final_page = 1
#pages = [str(p) for p in range(initial_page, final_page+1)]
pages = os.listdir(raw_dir)
total_faces = 0
for page in pages:
    page_dir = os.path.join(raw_dir, page)
    imgs = os.listdir(page_dir)
    for img in imgs:
        successful, new_face_count, out_file = detect(os.path.join(page_dir, img), page, os.path.join(cropped_dir, page))
        if successful:
            print('{} faces found and cropped: {}'.format(new_face_count, out_file))

    total_faces+=new_face_count          

    
