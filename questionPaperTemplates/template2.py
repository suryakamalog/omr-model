
#Still in production


import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import templates
import answersUtil as ans

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
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img

def showImg(image):
    image = ResizeWithAspectRatio(image, width=500)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def evaluate(image, thresh, questionCnts, idx):
    
    noOfQuestions = template["noOfQuestions"]
    ANSWER_KEY = ans.fetchAnswers(noOfQuestions)
    noOfOptions = template["noOfOptions"]
    colLength = template['questionBox_ColLength']
    rowLength = template['questionBox_RowLength']
    increment = idx*colLength
    
    questionCnts = imutils.contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    correct = 0
    
    for (q, i) in enumerate(np.arange(0, len(questionCnts), rowLength)):
        cnts = imutils.contours.sort_contours(questionCnts[i:i + rowLength])[0]
        
        for (j, c) in enumerate(np.arange(0, len(cnts), noOfOptions)):
            ansPos = q +j*colLength
            correctOption = ANSWER_KEY[ansPos+1+increment]
            color = (0, 0, 255)
            bubbled = None
            optionNum = 0
            mini = 1000
            maxi = 0
            for k in range(c,c+noOfOptions):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [cnts[k]], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                # showImg(mask)
                total = cv2.countNonZero(mask)
                # print(total)
                mini = min(total, mini)
                maxi = max(total, maxi)
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, optionNum)
                optionNum += 1
            if maxi - mini < 15:
                continue
            if correctOption == bubbled[1]:
                color = (0, 255, 0)
                correct += 1
                cv2.drawContours(image, [cnts[correctOption+c]], -1, color, 1)

    showImg(image)

    return correct

def findQuestionContours(image):
    # print("in findQuestionContours: ", image)
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
        if  ((w >= 11 and h >= 11) and (w <= 80 and h <= 80) and ar >= 0.7 and ar <= 1.5):
            questionCnts.append(currentContour)
            # these are the innermost child components
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)

    showImg(image)
    return (questionCnts, thresh)

def getDetailsFromContours(detailCnts, rowLength, colLength):
    limit = 125 #120px
    detailCnts = imutils.contours.sort_contours(detailCnts, method="top-to-bottom")[0]
    result = [0]*rowLength
    for (q, i) in enumerate(np.arange(0, len(detailCnts), rowLength)):
        cnts = imutils.contours.sort_contours(detailCnts[i:i + rowLength])[0]
        bubbled = []

        for (j, c) in enumerate(cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                if total > limit:
                    bubbled.append(j)

        for i in range(len(bubbled)):
            result[bubbled[i]] = chr(ord('A') + q)

    for i in range(len(result)):
        result_str += str(result[i])
    return result_str

def extractDetails(detailBoxes):

    studentNameBox, questionSetBox, rollNumberBox, subjectBox, mobileNumberBox, testIdBox = detailBoxes[0], detailBoxes[1], detailBoxes[2], detailBoxes[3], detailBoxes[4], detailBoxes[5]
    studentNameContours = findQuestionContours(warpAndCrop(studentNameBox, "detailBox"))
    questionSetContours = findQuestionContours(warpAndCrop(questionSetBox, "detailBox"))
    rollNumberContours = findQuestionContours(warpAndCrop(rollNumberBox, "detailBox"))
    subjectContours = findQuestionContours(warpAndCrop(subjectBox, "detailBox"))
    mobileNumberContours = findQuestionContours(warpAndCrop(mobileNumberBox, "detailBox"))
    testIdContours = findQuestionContours(warpAndCrop(testIdBox, "detailBox"))


    studentName = getDetailsFromContours(studentNameContours, template['nameLength'], 26)
    questionSet = getDetailsFromContours(questionSetContours, template['questionSetLength'], 4)
    rollNumber = getDetailsFromContours(rollNumberContours, template['rollNumberLength'], 10)
    subject = getDetailsFromContours(subjectContours, template['subjectLength'], 3)
    mobileNumber = getDetailsFromContours(mobileNumberContours, template['mobileNumberLength'], 10)
    testId = getDetailsFromContours(testIdContours, template['testIdLength'], 10)

    return (studentName, questionSet, rollNumber, subject, mobileNumber, testId)

def warpAndCrop(rect, rectType):

    x,y,w,h = rect
    four_points = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
    pts = np.array(four_points, dtype = "float32")
    # warped = imutils.perspective.four_point_transform(originalImage, pts)

    #Error: warped Image not giving appropriate result
    if rectType == "questionBox":
        cropped = originalImage[y+5:y+h-8, x+50:x+w-5] #cropping the image to remove any lines in border
    else :
        cropped = originalImage[y+90:y+h-8, x+5:x+w-5]
    showImg(cropped)
    return cropped

def extractBoxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    allRect = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            if h>200:
                allRect.append([x,y,w,h])

    firstLevel, secondLevel, thirdLevel = [],[],[]
    for rect in allRect:
        if rect[1] < 1000:
            if rect[1] < 600:
                firstLevel.append(rect)
            else:
                secondLevel.append(rect)
        else:
            thirdLevel.append(rect)


    firstLevel = sorted(firstLevel)
    studentName, questionSet, rollNumber, subject = firstLevel[0], firstLevel[1], firstLevel[2], firstLevel[3]
    secondLevel = sorted(secondLevel)
    mobileNumber, testId = secondLevel[0], secondLevel[1]
    thirdLevel = sorted(thirdLevel)
    questionBox1, questionBox2, questionBox3, questionBox4 = thirdLevel[0], thirdLevel[1], thirdLevel[2], thirdLevel[3]

    return [firstLevel + secondLevel, thirdLevel]

def scanBoxes(allQuestionBoxes):
    correct = 0
    idx = 0
    for questionBox in allQuestionBoxes:
        image = warpAndCrop(questionBox, "questionBox")
        questionCnts, thresh = findQuestionContours(image)
        correct += evaluate(image, thresh, questionCnts, idx)
        idx+=1
    return correct

def findCorners(image):
    img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    saveRect = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        if (w>10 and h>10) and (w<40 and h<40):
            saveRect.append([x,y,w,h])
            
    
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

    topLeft[0]+=35
    bottomLeft[0]+=35
    topRight[1]+=23
    bottomRight[1]-=20
    
    pts1 = [topLeft, topRight, bottomLeft, bottomRight]
    return pts1

def getBirdView(image):
    width, height = template['widthAfterTransform'], template['heightAfterTransform']
    pts1 = findCorners(image)
    pts1 = np.float32([pts1])
    pts2 = np.float32([[0,0], [width,0], [0,height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(image, matrix, (width, height))
    return imgOutput

def template_2_main(image, templateNum, answerFileName):
    global template, originalImage
    template = templates.getTemplateDetails(templateNum)
    image = getBirdView(image)
    showImg(image)
    originalImage = image.copy()
    detailBoxes, allQuestionBoxes = extractBoxes(image.copy())
    # details = extractDetails(detailBoxes)
    # studentName = details[0]
    # questionSet = details[1]
    # rollNumber = details[2]
    # subject = details[3]
    # mobileNumber = details[4]
    # testId = details[5]

    totalMarks = scanBoxes(allQuestionBoxes)

    print("Total Marks: ", totalMarks)
    # print("Student Name: ", studentName)
    # print("Question Set: ", questionSet)
    # print("Roll Number: ", rollNumber)
    # print("Subject: ", subject)
    # print("Question Set: ", mobileNumber)
    # print("Roll Number: ", testId)

    # return (studentName, questionSet, rollNumber, subject, mobileNumber, testId)
    return totalMarks