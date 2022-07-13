from tensorflow import keras
import os
import glob
import PIL
import PIL.Image as Image
import numpy as np

if __name__ == "__main__":
    model = keras.models.load_model("MY_CNN_Location.h5")

    test_FileName = "archive/seg_pred/seg_pred/"
    test_path = os.path.abspath("./"+test_FileName)
    test_files = glob.glob(test_path+"/*.jpg")

    actual_images = []
    temp_test_location_images = []

    for location_image in test_files:
        pil_im = Image.open(location_image).convert("RGB")
        actual_images.append(pil_im)
        img = np.asarray(pil_im).astype(np.float32)
        if(np.shape(img) == (150,150,3)):
            temp_test_location_images.append(img)



    n2 = len(temp_test_location_images)
    test_location_images = np.asarray(temp_test_location_images).astype(np.float32)
    test_location_images = test_location_images/ 255.0

    # 0 buildings , 1 forest, 2 glacier, 3 mountain, 4 sea, 5 street
    #print(np.shape(test_location_images[0]))
    set = []
    set.append(test_location_images[533:538]) # 333~337

    lis = model.predict(set)
    #print(model.predict(set))


    ans = {0 : " buildings", 1 : "forest", 2 : "glacier", 3 : "mountain", 4 : "sea", 5 : "street"}


    for i in lis:
        max = 0
        max_index = 0
        count = 0
        for k in i:
            if(max < k):
                max = k
                max_index = count
            count += 1
        print(ans[max_index])
    actual_images[533].show()
    actual_images[534].show()
    actual_images[535].show()
    actual_images[536].show()
    actual_images[537].show()
    #print(model.predict(set))
