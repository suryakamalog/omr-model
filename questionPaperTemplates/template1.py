import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
import answersUtil as ans
from imutils import contours
import templates
import sys
import json

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def scale(img, max_height, max_width):
    height, width = img.shape[:2]

    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width/float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor,
                         fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img


def showImg(image):
    image = ResizeWithAspectRatio(image, width=500)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def evaluate(image, thresh, questionCnts, whichRect):

    noOfQuestions = template["noOfQuestions"]
    ANSWER_KEY, ASSIGNED_MARKS, NEGATIVE_MARKS = ans.fetchAnswers(
        answerFileName, noOfQuestions)
    noOfOptions = template["noOfOptions"]
    colLength = None
    rowLength = None
    increment = 0
    if whichRect == "right":
        rowLength = template["rightBox_RowLength"]
        colLength = template["rightBox_ColLength"]
    elif whichRect == "left":
        rowLength = template["leftBox_RowLength"]
        colLength = template["leftBox_ColLength"]
        increment = 10

    questionCnts = imutils.contours.sort_contours(
        questionCnts, method="top-to-bottom")[0]
    correct = 0
    total_marks = 0
    maskCount = 0
    for (q, i) in enumerate(np.arange(0, len(questionCnts), rowLength)):
        cnts = imutils.contours.sort_contours(questionCnts[i:i + rowLength])[0]

        for (j, c) in enumerate(np.arange(0, len(cnts), noOfOptions)):
            ansPos = q + j*colLength
            correctOption = ANSWER_KEY[ansPos+1+increment]
            marks = ASSIGNED_MARKS[ansPos+1+increment]
            if NEGATIVE_MARKS[ansPos+1+increment] is None:
                negMarks = 0
            else:
                negMarks = NEGATIVE_MARKS[ansPos+1+increment]

            greenColor = (0, 255, 0)
            redColor = (0, 0, 255)
            bubbled = None
            optionNum = 0
            mini = 1000
            maxi = 0
            try:
                for k in range(c, c+noOfOptions):
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [cnts[k]], -1, 255, -1)
                    mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                    maskCount += 1
                    total = cv2.countNonZero(mask)
                    mini = min(total, mini)
                    maxi = max(total, maxi)
                    if bubbled is None or total > bubbled[0]:
                        bubbled = (total, optionNum)
                    optionNum += 1
            except:
                # bubbled1 = (0,0)
                # bubbled1[1] = None
                correctedAnswerSheet[str(ansPos+1+increment)] = {}
                correctedAnswerSheet[str(ansPos+1+increment)]["correctOption"] = str(correctOption)
                correctedAnswerSheet[str(ansPos+1+increment)]["selectedOption"] = "Invalid"
                correctedAnswerSheet[str(ansPos+1+increment)]["negativeMarks"] = str(negMarks)
                correctedAnswerSheet[str(ansPos+1+increment)]["marksAwarded"] = "0"
                correctedAnswerSheet[str(ansPos+1+increment)]["answerOutcome"] = "Invalid"
                continue

            correctedAnswerSheet[str(ansPos+1+increment)] = {}
            correctedAnswerSheet[str(ansPos+1+increment)]["correctOption"] = str(correctOption)
            correctedAnswerSheet[str(ansPos+1+increment)]["selectedOption"] = str(bubbled[1])
            correctedAnswerSheet[str(ansPos+1+increment)]["negativeMarks"] = str(negMarks)

            if maxi - mini < 15:
                correctedAnswerSheet[str(ansPos+1+increment)]["marksAwarded"] = "0"
                correctedAnswerSheet[str(ansPos+1+increment)]["answerOutcome"] = "Not Answered"
                


            elif correctOption == bubbled[1]:
                correct += 1
                total_marks += marks
                correctedAnswerSheet[str(ansPos+1+increment)]["marksAwarded"] = str(marks)
                correctedAnswerSheet[str(ansPos+1+increment)]["answerOutcome"] = "Correct"
                # correctedAnswerSheet[ansPos+1+increment] = "Correct (" + "+" + str(marks) + " marks)"
                cv2.drawContours(image, [cnts[correctOption+c]], -1, greenColor, 1)
            else:
                total_marks -= negMarks
                correctedAnswerSheet[str(ansPos+1+increment)]["marksAwarded"] = "-" + str(negMarks)
                correctedAnswerSheet[str(ansPos+1+increment)]["answerOutcome"] = "Incorrect"
                cv2.drawContours(image, [cnts[bubbled[1]+c]], -1, redColor, 1)
    # showImg(image)
    return (correct, total_marks)

def findQuestionContours(image,sno):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    items = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cnts = items[0]
    hierarchy = items[1]

    hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions

    questionCnts = []
    # For each contour, find the bounding rectangle and draw it
    for component in zip(cnts, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        x,y,w,h = cv2.boundingRect(currentContour)
        ar = w / float(h)
        if ((w >= 11 and h >= 11) and (w <= 70 and h <= 70)):
            questionCnts.append(currentContour)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)

    return (questionCnts, thresh)

def extractDetails(candidateDetailsBox):
    # showImg(candidateDetailsBox)
    detailCnts, thresh = findQuestionContours(candidateDetailsBox,"3")
    enrollmentCodeLength = template["enrollmentCodeLength"]
    testIdLength = template["testIdLength"]
    rowLength = enrollmentCodeLength + testIdLength
    colLength = 10
    limit = 125 #125px
    
    detailCnts = imutils.contours.sort_contours(detailCnts, method="top-to-bottom")[0]
    correct = 0
    enrollmentCode = [0]*10
    testId = [0]*5
    for (q, i) in enumerate(np.arange(0, len(detailCnts), rowLength)):
        cnts = imutils.contours.sort_contours(detailCnts[i:i + rowLength])[0]
        bubbled1, bubbled2 = [], []
        for (j, c) in enumerate(cnts):

            if j < 10:
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                if total > limit:
                    bubbled1.append(j)
            else:
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                if total > limit:
                    bubbled2.append(j-10)

        for i in range(len(bubbled1)):
            enrollmentCode[bubbled1[i]] = q
        for i in range(len(bubbled2)):
            testId[bubbled2[i]] = q

    enrollmentCode_str, testId_str = "", ""
    for i in range(len(enrollmentCode)):
        enrollmentCode[i] = (enrollmentCode[i]+1)%10
        enrollmentCode_str += str(enrollmentCode[i]) 
    for i in range(len(testId)):
        testId[i] = (testId[i]+1)%10
        testId_str += str(testId[i])
    
    return (enrollmentCode_str, testId_str) 

def warpAndCrop(rect, whichRect):

    x,y,w,h = rect
    four_points = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
    pts = np.array(four_points, dtype = "float32")
    warped = imutils.perspective.four_point_transform(originalImage, pts)

    # Error: warped Image not giving appropriate result
    if whichRect == "right":
        cropped = originalImage[y+60:y+h-10, x+10:x+w-10] #cropping the image to remove any lines in border
        cropped = scale(cropped, 185, 255)
    elif whichRect == "left":
        cropped = originalImage[y+32:y+h-10, x+10:x+w-10]
    elif whichRect == "topLeft":
        cropped = originalImage[y+60:y+h-10, x+5:x+w-10]

    # showImg(cropped)
    return cropped

def extractBoxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    rectangles = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            if w>100 and h>100:
                rectangles.append([x,y,w,h])

    # minimum size of the list should be 3
    Ymax, Xmax, minSum = 0,0,100000
    leftRect, rightRect, topLeftRect = [],[],[]
    for rect in rectangles:
        if rect[1] > Ymax:
            Ymax = rect[1]
            leftRect = rect
        if rect[0] > Xmax:
            Xmax = rect[0]
            rightRect = rect
        if rect[0]+rect[1] < minSum:
            minSum = rect[0]+rect[1]
            topLeftRect = rect


    leftQuestionBox = warpAndCrop(leftRect, "left")
    rightQuestionBox = warpAndCrop(rightRect, "right")
    candidateDetailsBox = warpAndCrop(topLeftRect, "topLeft")
    return (leftQuestionBox, rightQuestionBox, candidateDetailsBox)

def scanBoxes(image1, image2):
    questionCnts, thresh = findQuestionContours(image1,"1")
    correct1, totalMarks1 = evaluate(image1, thresh, questionCnts, "right")
    questionCnts, thresh = findQuestionContours(image2,"2")
    correct2, totalMarks2 = evaluate(image2, thresh, questionCnts, "left")
    return (correct1 + correct2, totalMarks1 + totalMarks2) 

def findCorners(image):
    fpImage = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    saveRect = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        if (w>5 and h>5) and (w<10 and h<10):
            saveRect.append([x,y,w,h])
            cv2.rectangle(fpImage,(x,y),(x+w,y+h),(0,0,255),1)
    
    pts1 = []
    for rect in saveRect:
        pts1.append(rect[:2])
    topLeft = None
    bottomRight = None
    topRight = None
    bottomLeft = None
    mini = 100000
    maxi = 0
    for i in pts1:
        if i[0] + i[1] < mini:
            mini = i[0]+i[1]
            topLeft = i
        if i[0] + i[1] > maxi:
            maxi = i[0]+i[1]
            bottomRight = i
    mini = 100000
    maxi = 0
    for i in pts1:
        if i[0] - i[1] < mini:
            mini = i[0]-i[1]
            bottomLeft = i
        if i[0] - i[1] > maxi:
            maxi = i[0] - i[1]
            topRight = i

    topRight[0]+=10
    bottomRight[0]+=10
    pts1 = [topLeft, topRight, bottomLeft, bottomRight]
    # print(pts1)
    return pts1

def getBirdView(image):
    width, height = template['widthAfterTransform'], template['heightAfterTransform']
    pts1 = findCorners(image)
    pts1 = np.float32([pts1])
    pts2 = np.float32([[0,0], [width,0], [0,height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # put in try 
    imgOutput = cv2.warpPerspective(image, matrix, (width, height)) #put in try
    return imgOutput

def template_1_main(image, templateNum, answer="answer1.xlsx"):

    global template, originalImage, correctedAnswerSheet, answerFileName
    
    answerFileName = answer
    correctedAnswerSheet = {}
    template = templates.getTemplateDetails(templateNum)
    image = getBirdView(image)
    originalImage = image.copy()
    leftQuestionBox, rightQuestionBox, candidateDetailsBox = extractBoxes(image.copy())
    details = extractDetails(candidateDetailsBox)
    enrollmentCode = details[0]
    testId = details[1]

    correctedAnswerSheet["roll"] = str(enrollmentCode)
    correctedAnswerSheet["testId"] = str(testId)
    totalCorrect, totalMarks = scanBoxes(rightQuestionBox, leftQuestionBox)

    print("Enrollment Code: ", enrollmentCode)
    print("Test ID: ", testId)
    print("Total Correct: ", totalCorrect)
    print("Total Marks: ", totalMarks)
    print(json.dumps(correctedAnswerSheet, indent = 4))
    return (correctedAnswerSheet)
