import cv2
import os.path

cropped_dir = 'cropped_images'
raw_dir = './raw_images/'

#websites = ['zerochan']
if not os.path.isdir(cropped_dir):
    os.mkdir(cropped_dir)

cropped_dir = './cropped_images/'


def detect(filename, page, site, out_dir, cascade_file = '../lbpcascade_animeface.xml'):
    """
    Detects faces in the specified file.
        Parameters:
            filename (str) : filename to detect faces in
            page, site (str) : used in naming
            out_dir (str) : output directory
            cascade_file (str) : cascade xml file to use with OpenCV
        Returns:
            (bool) : if faces were outputted
            (int) : how many faces were outputted
            (str) : outfile 
    """
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found " % cascade_file)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    # apply cascade as demonstrated in example
    cascade = cv2.CascadeClassifier(cascade_file)

    try:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(gray,
                                        # detector options
                                        scaleFactor = 1.1,
                                        minNeighbors = 5,
                                        minSize = (24, 24))
        face_count = len(faces)
        if len(faces) > 0:
            for face in range(len(faces)):
                cur_image = filename[-6:-4]
                x, y, w, h = faces[face]
                # crop face
                out_file = os.path.join(out_dir, '{}_{}_{}_{}.jpg'.format(site, page, cur_image, str(face+1).zfill(2)))
                cropped_image = image[int(y-0.1*h): int(y+0.9*h), x: x+w]
                cropped_h, cropped_w, cropped_c = cropped_image.shape
                if(cropped_h >= 64 and cropped_w >= 64):
                    cv2.imwrite(out_file, cropped_image)
                else:
                    print("didn't output face because too small")
                    face_count -= 1
                
            return True, face_count, out_file
        else:
            return False, face_count, ''
    except:
        print("problem in loading image: ", filename)
        return False, 0, ''


#initial_page = 1
#final_page = 1
#pages = [str(p) for p in range(initial_page, final_page+1)]
def run_detect_ZC():
    total_faces = 0
    site = 'zerochan'
    print('currently processing: ', site)
    cur_dir = os.path.join(raw_dir, site)
    pages = os.listdir(cur_dir)
    for page in pages:
        page_dir = os.path.join(cur_dir, page)
        imgs = os.listdir(page_dir)
        for img in imgs:
            successful, new_face_count, out_file = detect(os.path.join(page_dir, img), page, site, os.path.join(cropped_dir, site))
            if successful:
                print('{} faces found and cropped: {}'.format(new_face_count, out_file))
            
            total_faces+=new_face_count

    print("Total zerochan faces exported: ", total_faces)
    return total_faces

def run_detect_SS():
    total_faces = 0
    site = 'shuushuu'
    print('currently processing: ', site)
    
    cur_dir = os.path.join(raw_dir, site)
    categories = os.listdir(cur_dir)
    for category in categories:
        category_dir = os.path.join(cur_dir, category)
        pages = os.listdir(category_dir)
        for page in pages:
            page_dir = os.path.join(category_dir, page)
            imgs = os.listdir(page_dir)
            for img in imgs:
                successful, new_face_count, out_file = detect(os.path.join(page_dir, img), page, site, os.path.join(cropped_dir, site))
                if successful:
                    print('{} faces found and cropped: {}'.format(new_face_count, out_file))
                
                total_faces+=new_face_count

    print("Total shuushuu faces exported: ", total_faces)
    return total_faces

zc_faces = run_detect_ZC() 
ss_faces = run_detect_SS()

print("Total faces exported: ", zc_faces + ss_faces)
         
