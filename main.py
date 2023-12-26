from roboflow import Roboflow
import cv2
#from PIL import Image
img1 = 'SeedsImages/Bad_Img.jpg'

img2 = 'SeedsImages/Good_Img.jpg'
#img2_use_path = img2.copy()
img3 = 'SeedsImages/TestImg.jpg'
#img3_use_path = img3.copy()

def GetPercentOfGoodSeeds(img_path):
    rf = Roboflow(api_key="6kFW5Dwy6q6IijhvfbT9")
    project = rf.workspace().project("wheat_rating")
    model = project.version(1).model
    predictions = model.predict(img_path, confidence=40, overlap=30).json()
    objs = predictions.values()
    number_of_good_seeds = 0
    total_number_of_seeds = 0
    for i in objs:
        for j in i:
            total_number_of_seeds += 1
            if type(j) == dict:
                if j['class'] == 'good':
                    number_of_good_seeds += 1
    PercentOfGoodSeeds = round((number_of_good_seeds/total_number_of_seeds)*100,2)
    return PercentOfGoodSeeds
def ShowVisualModel(img_path):
    rf = Roboflow(api_key="6kFW5Dwy6q6IijhvfbT9")
    project = rf.workspace().project("wheat_rating")
    model = project.version(1).model
    model.predict(img_path, confidence=40, overlap=30).save("Result_pic.jpg")


print(GetPercentOfGoodSeeds(img2), "% +- 10%", ShowVisualModel(img2))




    #PercentOfGoodSeeds = (number_of_good_seeds/total_number_of_seeds)*100

   # return PercentOfGoodSeeds



   # total_objects = len(predictions)
    #good_objects_count = sum(1 for pred in predictions if pred['class'] == 'good')
    #good_seeds_percent = (good_objects_count/total_objects)*100



#print(GetPercentOfGoodSeeds(img1))





#cv2.imread(img)
#cv2.imshow("result", img)
#cv2.waitKey(0)