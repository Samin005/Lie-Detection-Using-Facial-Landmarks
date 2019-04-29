import os.path
dataFile = 'truthData.csv'
if not os.path.exists(dataFile):
    excelFile = open(dataFile, 'w')
    excelFile.write("ID, Name, Sex \n")

    students = [['1', 'Aaron', "Male"], ['2', 'Bella', "Female"], ['3', 'Chambers', "Male"]]
    for s in students:
        excelFile.write(s[0] + ", " + s[1] + ", " + s[2] + "\n")
    excelFile.close()
