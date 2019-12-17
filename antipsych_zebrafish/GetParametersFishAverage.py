# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:43:48 2018

@author: Computer
"""

from mysql.connector import MySQLConnection, Error #Too be able to set up a connection to the local MySQL database running on a docker VM.
from python_mysql_dbconfig import read_db_config  #Config file is located in root folder with connection data to local database.
import random #Too generate the 50 parameters
import os #Too get folder content
import re
import sys
import numpy as np
import pickle
import operator
import progressbar
#from tqdm import tqdm



pathToExcel = 'J:\Kombinationsexperiment' #Specify where the combinational data is stored.

printMessages = False

def getOneSetOfParameters():
    dbconfig = read_db_config()
    conn = MySQLConnection(**dbconfig) 
    cursor = conn.cursor() #Create a cursor which runs the MySQL commands
    setOfParameters = [] #The final combined behavourial parameters of antipsych. + (antipsych+PTZ)
    setOfConcentrations = [] #The vector which decribes the input concentrations
    randomExcelOrMySQL = random.randint(1,100) # 1=Excel, 2=MySQL, pick a random place to fetch data
    antiPsychList = ['Aripiprazole', #The order of input concentrations OBS! Remember that an extra input is added which is called control which allways has the value 1.
'clozapine',
'Cariprazine',
'Haloperidol',
'Pimavanserin',
'Risperidone',
'NDMC',
'PCAP1',
'PCAP2',
'PCAP021',
'PCAP831',
'PCAP814',
'PCAP931',
'MePCAP1',
'CP809101',
'Basimglurant',
'MDL100907',
'Nelotanserin',
'OSU6162',
'SB242084',
'ChlDHA',
'Dipraglurant',
'DOI',
'MK801',
'PTZ',
'Quinpirole',
'CNO',
'DKB627',
'Hydroxynorketamine',
'Xanomeline',
'TS147.2',
'Quetiapine',
'Pridopidine',
'Midazolam',
'Lurasidone',
'KE091',
'Brexpiprazole',
'Biperiden',
'ADZ2066',
'8OHDPAT'] 
#    apl = np.asarray(antiPsychList)
#    np.save('Antipsychlist',apl)
    concentration = [0 for number in range(len(antiPsychList)+1)] #Set the concentration vector which is equal to the number of substances + 1 parameter for the control fish. which hopefully catches the "normal" behaviour.
                                                                    
           
    
    parameterNamesOriginal = open(os.path.join(sys.path[0], 'parameterNamesDatabaseExport.txt')).read().split(", "); # Read the chosen output behavioural parameters
    
   

    
    if(randomExcelOrMySQL < 11):
        if(printMessages==True):print("Excel")
        antiPsychDictCombo = {'Ari' : "Aripiprazole", "Cloz" : "clozapine", "Halo" : "Haloperidol", "Pim":"Pimavanserin", "Risp":"Risperidone"} #In excel the substances are named like this.
        
        randomFolder = random.randint(0,len(os.listdir(pathToExcel))-1) #Pick a random folder
        
        fileName = [] #.csv file names
        
        for file in os.listdir(pathToExcel + "\\" + os.listdir(pathToExcel)[randomFolder]): #List the two .csv files in the folder
                if file.endswith(".csv"):
                   fileName.append(file) 
                
        
        
        ### Import parameters for individual with antipsychotic
        if(fileName[0].endswith("light dark.csv")):
            rawData = open(pathToExcel + "\\" + os.listdir(pathToExcel)[randomFolder] + "\\" + fileName[0]).read()
            if(printMessages==True):print(pathToExcel + "\\" + os.listdir(pathToExcel)[randomFolder] + "\\" + fileName[0])
        else:
            rawData = open(pathToExcel + "\\" + os.listdir(pathToExcel)[randomFolder] + "\\" + fileName[1]).read()
            if(printMessages==True):print(pathToExcel + "\\" + os.listdir(pathToExcel)[randomFolder] + "\\" + fileName[1])
        
        
        rawData = rawData.split('\n')
        
        titleBar = rawData[0] #Create a vector with Excel parameternames
        titleBar = titleBar.split(';')
#        del titleBar[len(titleBar)-26:len(titleBar)]
        del titleBar[0] #Remove the four first columns
        del titleBar[0]
        del titleBar[0]
        del titleBar[0]
       
        
        
        randomIndividual = random.randint(1,len(rawData)-2) #Row 1 is a titlebar, the last row is empty in the .csv file.
        dataString = rawData[randomIndividual].split(';') #Take the random individual and split the data into a vector
        allIndividualData = [d.split(';') for d in rawData]
        del allIndividualData[len(allIndividualData)-1:len(allIndividualData)] #Remove the empty last line
        del allIndividualData[0] #Delete the titlebar
        
        del dataString[len(dataString)-26:len(dataString)] #Remove the parameters Quick# and Sumquick#
        
        typeOfFish = dataString[1]
   
        if(dataString[1] == 'Control'): #If controlfish set all input concentrations to 0 and the last parameter to 1
            concentration[len(concentration)-1] = 1  
            if(printMessages==True):print("Controlfish")
        else:
            concentration[len(concentration)-1] = 1 #The last input will allways be 1
            concentrationName = dataString[1].split(' ')
            if(printMessages==True):print(concentrationName)
            match = re.match(r"([a-z]+)([0-9]+)", concentrationName[0], re.I) #Splits the, for example, Ari3 into Ari and 3
            if match:
                items = match.groups()
                
                
            for n, i in enumerate(antiPsychList): #Add the first concentration to concentration vector
               if i == antiPsychDictCombo.get(items[0]):
                   concentration[n] = float(items[1])
                   
            
            
            match = re.match(r"([a-z]+)([0-9]+)", concentrationName[1], re.I) #Splits the, for example, Ari3 into Ari and 3
            if match:
                items = match.groups()
                
            for n, i in enumerate(antiPsychList): #Add the second concentration to the vector
               if i == antiPsychDictCombo.get(items[0]):
                   concentration[n] = float(items[1])
        
        
        del dataString[0] #Remove the four first columns because they are non-behavioural parameters
        del dataString[0]
        del dataString[0]
        del dataString[0]
           
        #Check which lines have the same treatment
        sameTreatInd = []
        for i, v in enumerate(allIndividualData):
            if(v[1] == typeOfFish):
                sameTreatInd.append(i)
            del allIndividualData[i][0] #Remove the four first columns because they are non-behavioural parameters
            del allIndividualData[i][0] 
            del allIndividualData[i][0] 
            del allIndividualData[i][0]
#            del allIndividualData[i][len(allIndividualData[i])-26:len(allIndividualData[i])] #Remove the last parameters sumquick and quick
            for j, d in enumerate(v): #Find empty strings which are unconvertable and replace with 0
                if d == '':
                    allIndividualData[i][j] = '0'
                allIndividualData[i][j] = float(allIndividualData[i][j])   
            
        for i,v in enumerate(allIndividualData):
            for j, d in enumerate(v):
                if d > 60000:
                    allIndividualData[i][j] = 0
            
        #Remove the first rows, the last ones and replace '' with zeros
        
        
       
        for n, i in enumerate(dataString): #Find empty strings which are unconvertable and replace with 0
            if i == '':
                dataString[n] = '0'
                
        
        
        dataString = [float(x) for x in dataString] #convert into float
        inputAntipsych = [0.0 for x in range(len(parameterNamesOriginal))]
        sumParam = 0
        numberOfInd = 0
        for idx, checkParameter in enumerate(parameterNamesOriginal): #Match the chosen behavioural parameters with the ones in the excel sheet.
            for j, excelParameter in enumerate(titleBar):
                if (checkParameter == excelParameter):
                    for i, v in enumerate(sameTreatInd):
                        if allIndividualData[v][j] != 0:
                            sumParam += allIndividualData[v][j]
                            numberOfInd +=1
                    if(numberOfInd != 0):inputAntipsych[idx] = (sumParam/numberOfInd)
                    else:inputAntipsych[idx]=(0.0)
                    sumParam = 0
                    numberOfInd = 0
                    break
        
            
                
        if(printMessages==True):print("LENGTH OF LIST1 "+str(len(inputAntipsych)))
            
                    
        
        
        ### Import parameters for individual with antipsychotic+PTZ
        if(fileName[0].endswith("PTZ dark.csv")):
            rawData = open(pathToExcel + "\\" + os.listdir(pathToExcel)[randomFolder] + "\\" + fileName[0]).read()
            if(printMessages==True):print(pathToExcel + "\\" + os.listdir(pathToExcel)[randomFolder] + "\\" + fileName[0])
        else:
            rawData = open(pathToExcel + "\\" + os.listdir(pathToExcel)[randomFolder] + "\\" + fileName[1]).read()
            if(printMessages==True):print(pathToExcel + "\\" + os.listdir(pathToExcel)[randomFolder] + "\\" + fileName[1])
        
        rawData = rawData.split('\n')
        
        titleBar = rawData[0] #Create a vector with Excel parameternames
        titleBar = titleBar.split(';')
#        del titleBar[len(titleBar)-26:len(titleBar)]
        del titleBar[0] #Remove the four first columns
        del titleBar[0]
        del titleBar[0]
        del titleBar[0]
       
        
        
        randomIndividual = random.randint(1,len(rawData)-2) #Row 1 is a titlebar, the last row is empty in the .csv file.
        dataStringPTZ = rawData[randomIndividual].split(';') #Take the random individual and split the data into a vector
        allIndividualData = [d.split(';') for d in rawData]
        del allIndividualData[len(allIndividualData)-1:len(allIndividualData)] #Remove the empty last line
        del allIndividualData[0] #Delete the titlebar
        
        del dataStringPTZ[len(dataStringPTZ)-26:len(dataStringPTZ)] #Remove the parameters Quick# and Sumquick#
        
        typeOfFish = dataStringPTZ[1]
        
        del dataStringPTZ[0] #Remove the four first columns because they are non-behavioural parameters
        del dataStringPTZ[0]
        del dataStringPTZ[0]
        del dataStringPTZ[0]
           
        #Check which lines have the same treatment
        sameTreatInd = []
        for i, v in enumerate(allIndividualData):
            if(v[1] == typeOfFish):
                sameTreatInd.append(i)
            del allIndividualData[i][0] #Remove the four first columns because they are non-behavioural parameters
            del allIndividualData[i][0] 
            del allIndividualData[i][0] 
            del allIndividualData[i][0]
#            del allIndividualData[i][len(allIndividualData[i])-26:len(allIndividualData[i])] #Remove the last parameters sumquick and quick
            for j, d in enumerate(v): #Find empty strings which are unconvertable and replace with 0
                if d == '':
                    allIndividualData[i][j] = '0'
                allIndividualData[i][j] = float(allIndividualData[i][j])   
            
        for i,v in enumerate(allIndividualData):
            for j, d in enumerate(v):
                if d > 60000:
                    allIndividualData[i][j] = 0
            
        #Remove the first rows, the last ones and replace '' with zeros
        
        
       
        for n, i in enumerate(dataString): #Find empty strings which are unconvertable and replace with 0
            if i == '':
                dataStringPTZ[n] = '0'
                
        
        
#        dataStringPTZ = [float(x) for x in dataStringPTZ] #convert into float
        inputAntipsychPTZ = [0.0 for x in range(len(parameterNamesOriginal))]
        sumParam = 0
        numberOfInd = 0
        for idx, checkParameter in enumerate(parameterNamesOriginal): #Match the chosen behavioural parameters with the ones in the excel sheet.
            for j, excelParameter in enumerate(titleBar):
                if (checkParameter == excelParameter):
                    for i, v in enumerate(sameTreatInd):
                        if allIndividualData[v][j] != 0:
                            sumParam += allIndividualData[v][j]
                            numberOfInd +=1
                    if(numberOfInd != 0):inputAntipsychPTZ[idx] = (sumParam/numberOfInd)
                    else:inputAntipsychPTZ[idx]=(0.0)
                    sumParam = 0
                    numberOfInd = 0
                    break
                
        if(printMessages==True):print("LENGTH OF LIST2 "+str(len(inputAntipsychPTZ)))
        
        
   
            
                    
                   
        
        if(printMessages==True):print(concentration)
        dataCombined = inputAntipsych + inputAntipsychPTZ
        setOfParameters = dataCombined    #Add the output: combined behavourial parameters for antipsych. and antipsych.+PTZ to the matrix
        
        setOfConcentrations = concentration #Add the input concentrations to the matrix
       
        
      
    else:
       if(printMessages==True):print('MySQL')
       parameterNameList = [] #Define the list of variable names in database
       parameterNameListPtz = [] #The list of variable names of experiments with antipsych.+ptz in database
       while len(parameterNameList) == 0 or len(parameterNameListPtz) == 0: #Master condition for finding parameters, none of the output parameter sets can be empty
           parameterNameList = [] #Set vector to empty for every new try
           parameterNameListPtz = [] #Set vector to empty for every new try
           dataString = []
           ##Pick a random experiment
           cursor.execute("SELECT id, assay from experiments where experiments.assay = 'lightdark' and experiments.substance != 'PCAP831andTrospium' or experiments.assay = 'ptz' and experiments.substance != 'PCAP831andTrospium'")
           experimentIdList = []
           assayList = []
           substanceList = []
           for item in cursor:
               experimentIdList.append(item[0]) #List all the experiments
               
           cursor.execute("SELECT id, assay from experiments where experiments.assay = 'lightdark' or experiments.assay = 'ptz'")    
           for item in cursor:
               assayList.append(item[1]) #List all the assays
           cursor.execute("SELECT id, assay, substance from experiments where experiments.assay = 'lightdark' or experiments.assay = 'ptz'")    
           for item in cursor:
               substanceList.append(item[2]) #List all the substances
               
               
               
           
           parameterValues = [] #Save the parameter values
           individualIdList = []
           inexpIdList = []
    #       while individualIdList == []: #Some individuals doesnt have any parameters therefore I loop until we find some.
           randExperimentId = random.randint(0,len(experimentIdList)-1)
               
           if(printMessages==True):print(substanceList[randExperimentId], assayList[randExperimentId])
           cursor.execute("SELECT id, inexp_id from individuals where individuals.exp_id = "+str(experimentIdList[randExperimentId]))
               
           for item in cursor:
                   individualIdList.append(item[0])
                   
                
           cursor.execute("SELECT id, inexp_id from individuals where individuals.exp_id = "+str(experimentIdList[randExperimentId]))     
           for item in cursor:
                   inexpIdList.append(item[1])
                   
           
           
           
           
           randIndividualId = random.randint(0,len(individualIdList)-1)
           treatment = []
           cursor.execute("SELECT * from individuals where individuals.id = "+str(individualIdList[randIndividualId]))
           for item in cursor:
               treatment.append(item[2])
#           print(treatment)
           sameTreatmentId = []
           cursor.execute("SELECT id from individuals where individuals.exp_id ="+str(experimentIdList[randExperimentId])+" and individuals.treatment = '"+treatment[0]+"'")
           for item in cursor:
               sameTreatmentId.append(item[0])
           
           
           numberOfInd = [0.0 for x in range(len(parameterNamesOriginal))]
           sumParam = [0.0 for x in range(len(parameterNamesOriginal))]
           for i, v in enumerate(sameTreatmentId):
               parameterNameList = []
               parameterValues = []
           
               cursor.execute("SELECT name from datapoints where datapoints.individual_id = "+str(v))
               
               for item in cursor:
                   parameterNameList.append(item[0])
                       
        #         
                       
               cursor.execute("SELECT value from datapoints where datapoints.individual_id = "+str(v)+";")
               for item in cursor:
                       parameterValues.append(item[0])
                       
                  
               
               
               sumParameters = 0
               iteration = 0
               paramNum = 0
               for idx, checkParameter in enumerate(parameterNamesOriginal):
                    
                    for j, MySQLParameters in enumerate(parameterNameList):
                        if (checkParameter == MySQLParameters):
                            if(parameterValues[j] == 0 or parameterValues[j] > 60000):
                                if(printMessages==True):print("")
                            else:
                                sumParam[idx] += parameterValues[j]
                                numberOfInd[idx] += 1
                            
                            break
           input1 = [0.0 for x in range(len(parameterNamesOriginal))]
           for i,v in enumerate(sumParam):
               if(v != 0):input1[i]=v/numberOfInd[i]
                 
           if(printMessages==True):print("LENGTH OF LIST1MYSQL "+str(len(input1)))
           if(printMessages==True):print(input1)
           manualAverage = 0
           for i in range(13):
               manualAverage += parameterValues[i]
           if(printMessages==True):print(manualAverage/13)
           
          
           if substanceList[randExperimentId] != 'PTZ high': #Because PTZ as substance doesnt have an other experiment with assay ptz
               secondExperimentIdList = []
               
               if(assayList[randExperimentId] == 'ptz'):
                   cursor.execute("SELECT * from experiments where experiments.assay = 'lightdark' and experiments.substance = '"+substanceList[randExperimentId]+"'")
                   if(printMessages==True):print(substanceList[randExperimentId], "lightdark")
               else:
                   cursor.execute("SELECT * from experiments where experiments.assay = 'ptz' and experiments.substance = '"+substanceList[randExperimentId]+"'")
                   if(printMessages==True):print(substanceList[randExperimentId], "ptz")
               for item in cursor:
                   secondExperimentIdList.append(item[0]) 
               
               
               
               idList = []
               
               cursor.execute("SELECT id, inexp_id from individuals where individuals.exp_id = "+str(secondExperimentIdList[0])+" and inexp_id = "+str(inexpIdList[randIndividualId]))     
               for item in cursor:
                       idList.append(item[0])
                      
               parameterValues = []
               
               cursor.execute("SELECT value from datapoints where datapoints.individual_id = "+str(idList[0]))
               for item in cursor:
                       parameterValues.append(item[0])
               cursor.execute("SELECT name from datapoints where datapoints.individual_id = "+str(idList[0]))
               for item in cursor:
                       parameterNameListPtz.append(item[0])
               
               treatment = []
               cursor.execute("SELECT * from individuals where individuals.exp_id = "+str(secondExperimentIdList[0])+" and inexp_id = "+str(inexpIdList[randIndividualId])) 
               for item in cursor:
                   treatment.append(item[2])
#               print(treatment)
               sameTreatmentId = []
               cursor.execute("SELECT id from individuals where individuals.exp_id ="+str(secondExperimentIdList[0])+" and individuals.treatment = '"+treatment[0]+"'")
               for item in cursor:
                   sameTreatmentId.append(item[0])
               
               numberOfInd = [0.0 for x in range(len(parameterNamesOriginal))]
               sumParam = [0.0 for x in range(len(parameterNamesOriginal))]
               for i, v in enumerate(sameTreatmentId):
                   parameterNameList = []
                   parameterValues = []
               
                   cursor.execute("SELECT name from datapoints where datapoints.individual_id = "+str(v))
                   
                   for item in cursor:
                       parameterNameList.append(item[0])
                           
            #         
                           
                   cursor.execute("SELECT value from datapoints where datapoints.individual_id = "+str(v)+";")
                   for item in cursor:
                           parameterValues.append(item[0])
                           
                      
                   
                   
                   sumParameters = 0
                   iteration = 0
                   paramNum = 0
                   for idx, checkParameter in enumerate(parameterNamesOriginal):
                        
                        for j, MySQLParameters in enumerate(parameterNameList):
                            if (checkParameter == MySQLParameters):
                                if(parameterValues[j] == 0 or parameterValues[j] > 60000):
                                    if(printMessages==True):print("")
                                else:
                                    sumParam[idx] += parameterValues[j]
                                    numberOfInd[idx] += 1
                                
                                break
               input2 = [0.0 for x in range(len(parameterNamesOriginal))]
               for i,v in enumerate(sumParam):
                   if(v != 0):input2[i]=v/numberOfInd[i]
               
               if(assayList[randExperimentId] == 'ptz'):
                   dataCombined = input2 + input1
               else:
                   dataCombined = input1 + input2
               setOfParameters = dataCombined
               if(printMessages==True):print("mysql")
               if(printMessages==True):print(len(dataCombined))
               
           else: #If SUBSTANCE is PTZ then fetch all controlfishes with assay ptz and average them 
#            print("**********PTZ************")
            with open('FishAverageDatabaseExportControlFishesPTZList.txt', 'rb') as f: #The dump of average control fishes with assay ptz is stored in a binary file with pickle
             averageControlFishParameters = pickle.load(f)
             
           
            print(len(input1),len(averageControlFishParameters))
            dataCombined = input1 + averageControlFishParameters
            setOfParameters = dataCombined
#            print("AVERAGE")
#            print(len(dataCombined))
            break #break the master while loop
       if(printMessages==True):print(individualIdList)   
       cursor.execute("SELECT * from individuals where individuals.id = "+str(individualIdList[randIndividualId]))
       substanceInformation = []
       for item in cursor:
               substanceInformation.append(item[3])
       cursor.execute("SELECT * from individuals where individuals.id = "+str(individualIdList[randIndividualId]))
       concentrationInformation = []
       for item in cursor:
               concentrationInformation.append(item[4])
       if(substanceInformation[0] == 'Control'):
           if(printMessages==True):print("Controlfish")
           concentration[len(concentration)-1] = 1
       else:
           concentration[len(concentration)-1] = 1
           for n, i in enumerate(antiPsychList): #Add concentration to concentration vector
               if i == substanceInformation[0]:
                   concentration[n] = float(concentrationInformation[0])
       setOfConcentrations = concentration
       
       
    if(printMessages==True):print(concentration)
    
    for i, v in enumerate(setOfParameters):
        if(v > 60000):
            setOfParameters[i] = 0
            if(printMessages==True):print("*********************65532 VALUE!!!***********************")
#    print("LENGTH: ")        
#    print(len(setOfParameters), len(setOfConcentrations))
    if(printMessages==True):print(type(setOfParameters), type(setOfConcentrations))
    conn.close()
    return(setOfParameters, setOfConcentrations)
        
#
#
def get50SetsOfParameters(batchSize):
#    with open('timeAverageControlFishesPTZList.txt', 'rb') as f:
#       my_list = pickle.load(f)
#    my_list = my_list + my_list
#    file = open("DebugingData.txt","a")
#    file.write("***********************************************´n")
#    file.write("*********************NEW BATCH*****************\n")
#    file.write("***********************************************\n")  
    parameterMATRIX = []
    concentrationMATRIX = []
    for i in progressbar.progressbar(range(batchSize)):
        returnVector = getOneSetOfParameters()
        parameterMATRIX.append(returnVector[0])
        concentrationMATRIX.append(returnVector[1])
        if(printMessages==True):print("********************Number "+ str(i) +"*********************************\n")
#        file.write("***************PARAMETER "+str(i)+"****************\n")
#        if(printMessages==True):print(len(returnVector[0]),len(my_list))
#        file.write(str([v/my_list[i] for i,v in enumerate(returnVector[0])]))
#        file.write("\n")
    
#    print(len(parameterMATRIX))
#    print(len(parameterMATRIX[0]))
    if(printMessages==True):print(len(concentrationMATRIX))
    if(printMessages==True):print(len(concentrationMATRIX[0]))
#    for i in range(batchSize):
#        print(len(parameterMATRIX[i]))

    
     
    
    
#    file.close() 
#    
    parameterNp = np.asarray(parameterMATRIX)
    concentrationNp = np.asarray(concentrationMATRIX)
#    print(type(parameterNp),type(concentrationNp))
    
    return(parameterNp, concentrationNp)
    #return parameterMATRIX, concentrationMATRIX
def test(n):
    print("in test")
    return(n*n)
#    
#p, c = get50SetsOfParameters(10)
