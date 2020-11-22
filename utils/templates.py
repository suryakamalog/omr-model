
template1_Details = {
	"noOfQuestions" : 20,
	"noOfOptions" : 4,
	"rightBox_RowLength" : 8,
	"rightBox_ColLength" : 5,
	"leftBox_RowLength" : 20,
	"leftBox_ColLength" : 2,
	"enrollmentCodeLength" : 10,
	"testIdLength" : 5,
	"widthAfterTransform" : 713,
	"heightAfterTransform" : 474
}

template2_Details = {
	"noOfQuestions" : 120,
	"noOfOptions" : 4,
	"questionBox_RowLength" : 4,
	"questionBox_ColLength" : 30,
	"nameLength" : 30,
	"questionSetLength" : 1,
	"rollNumberLength" : 7,
	"subjectLength" : 3,
	"mobileNumber" : 10,
	"testIdLength" : 2,
	"widthAfterTransform" : 1600,
	"heightAfterTransform" : 2247
}
def getTemplateDetails(templateNum):
	if templateNum == 1:
		return template1_Details
	if templateNum == 2:
		return template2_Details