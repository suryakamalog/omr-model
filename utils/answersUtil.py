from openpyxl import load_workbook

def fetchAnswers(answerFileName, noOfQuestions):
	workbook = load_workbook(filename=answerFileName)
	sheet = workbook.active

	answers, assignedMarks, negativeMarks = dict(), dict(), dict()
	quesNum = 1

	for row in sheet["A1:A" + str(noOfQuestions)]:
		answers[quesNum] = ord(list(row[0].value)[0]) - ord('A')
		quesNum += 1

	quesNum = 1
	for row in sheet["B1:B" + str(noOfQuestions)]:
		assignedMarks[quesNum] = row[0].value
		quesNum += 1

	quesNum = 1
	for row in sheet["C1:C" + str(noOfQuestions)]:
		negativeMarks[quesNum] = row[0].value
		quesNum += 1

	return (answers, assignedMarks, negativeMarks)