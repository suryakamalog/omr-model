import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import cv2
import argparse
import sys
sys.path.insert(1,'./utils')
sys.path.insert(1,'./questionPaperTemplates')
import answersUtil as ans
import templates
from template1 import *
from template2 import *



def main(image, templateNum, answer="answer.xlsx"):

    # main function will receive
    # 1. templateNum
    # 2. image
    if templateNum == 1:
        return template_1_main(image, templateNum, answer)
    if templateNum == 2:
        return template_2_main(image, templateNum, answer)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    ap.add_argument("-t", "--template", type=int,
                    required=True, help="template number")
    ap.add_argument("-a", "--answer", required=True,
                    help="path to the answer file")
    args = vars(ap.parse_args())
    imageName = args['image']
    templateNum = args['template']
    answer = args['answer']
    image = cv2.imread(imageName)
    main(image=image, templateNum=templateNum, answer=answer)
