def readCsvFile(file):
    inp = {}
    with open(file, 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            inp[row[0]] = row[1]
    fd.close()
    return inp

def modCsvFile(inpFile,typeOfMod,parameterType):
    tmpFile = "tmp.csv"
    with open(inpFile, "r") as file, open(tmpFile, "w") as outFile:
        reader = csv.reader(file, delimiter=',')
        writer = csv.writer(outFile, delimiter=',')
        header = next(reader)
        writer.writerow(header)
        if typeOfMod == 'analysisDone':
            for row in reader:
                colValues = []
                if row[0] == 'rew':
                    colValues.append('rew')
                    colValues.append('1')
                elif row[0] == 'mob_patterns':
                    colValues.append('mob_patterns')
                    colValues.append('1')
                elif row[0] == 'new_demand_dist':
                    colValues.append('new_demand_dist')
                    colValues.append('1')
                else:
                    for col in row:
                        colValues.append(col.lower())
                writer.writerow(colValues)
        elif typeOfMod == 'resetRew':
            for row in reader:
                colValues = []
                if row[0] == 'rew':
                    colValues.append('rew')
                    colValues.append('0')
                else:
                    for col in row:
                        colValues.append(col.lower())
                writer.writerow(colValues)
        elif typeOfMod == 'resetMobilityPatterns':
            for row in reader:
                colValues = []
                if row[0] == 'mob_patterns':
                    colValues.append('mob_patterns')
                    colValues.append('0')
                else:
                    for col in row:
                        colValues.append(col.lower())
                writer.writerow(colValues)
        else:
            for row in reader:
                colValues = []
                if row[0] == parameterType:
                    colValues.append(parameterType)
                    colValues.append(typeOfMod)
                else:
                    for col in row:
                        colValues.append(col.lower())
                writer.writerow(colValues)
    os.rename(tmpFile, inpFile)