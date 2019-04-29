import os.path

print "1. Record Lie Data"
print "2. Record Truth Data"
print "3. Test Data"

x = input("Please enter value: ")
if x == 1:
    print "Recording Lie Data"
    dataFile = 'lieData.csv'
    if os.path.exists(dataFile):
        # print "Exists"
        excelFile = open(dataFile, 'a')
        excelFile.write("4, Daisy, Female \n")
        excelFile.close()
    else:
        # print "does not exist"
        excelFile = open(dataFile, 'w')
        excelFile.write("ID, Name, Sex \n")

        students = [['1', 'Aaron', "Male"], ['2', 'Bella', "Female"], ['3', 'Chambers', "Male"]]
        for s in students:
            excelFile.write(s[0] + ", " + s[1] + ", " + s[2] + "\n")
        excelFile.close()
elif x == 2:
    print "Recording Truth Data"
    dataFile = 'truthData.csv'
    if os.path.exists(dataFile):
        # print "Exists"
        excelFile = open(dataFile, 'a')
        # excelFile.write("4, Daisy, Female \n")
        shapes = [[5, 9], [8, 7], [0, 4]]
        isTrue = 0;
        shapesString = ""
        for x in range(0, 3):
            tempArr = shapes[x]
            shapesString += `tempArr[0]`+", "+`tempArr[1]`+", "+`isTrue`+"\n"
        print shapesString
        excelFile.write(shapesString)
        # excelFile.write("asdq \, qwd, t, y, i")
        excelFile.close()
    else:
        # print "does not exist"
        excelFile = open(dataFile, 'w')
        excelFile.write("ID, Name, Sex \n")

        students = [['1', 'Aaron', "Male"], ['2', 'Bella', "Female"], ['3', 'Chambers', "Male"]]
        for s in students:
            excelFile.write(s[0] + ", " + s[1] + ", " + s[2] + "\n")
        excelFile.close()
elif x == 3:
    print "Testing Data"

