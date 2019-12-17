 
#Method where i get data, this method is started in 8 separate processes
def fetch(q,n):
    from GetParametersFishAverage import get50SetsOfParameters
    while(True):
        q.put(get50SetsOfParameters(n))
        print("The queue size is now: "+str(q.qsize()))
def fetchEva(Eq,v):
    from GetParametersFishAverage import get50SetsOfParameters
    while(True):
        Eq.put(get50SetsOfParameters(v))
        print("The evaluation queue size is now: "+str(Eq.qsize()))

def countIndividuals(concvector,substancesCount,antiPsychList):
  import pickle
  for i,v in enumerate(concvector):
      for ind,val in enumerate(v[:-1]):
          if(val != 0):
              try:
                  substancesCount[antiPsychList[ind]][str(val)] += 1
              except:
                  print("Found concentration which is not in the dict!")
  with open("substanceCount.pkl", 'wb') as f:
      pickle.dump(substancesCount, f, pickle.HIGHEST_PROTOCOL)
#def load_obj(name ):
#    with open('obj/' + name + '.pkl', 'rb') as f:
#        return pickle.load(f)




if __name__ == "__main__":
    
    from keras.models import Sequential, Model
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import *
    from keras.optimizers import SGD, RMSprop, Adam
    from keras.models import load_model
    from keras import backend as K
    from keras.layers import Reshape, LeakyReLU, BatchNormalization, Input, Concatenate
    from keras.initializers import RandomUniform
    from keras.utils import plot_model#plot_model(model) plt.plot(model.predict(concentrationNpEva));plt.plot(parameterNpEva);
    import numpy
    import numpy as np
    import math
    import matplotlib.pyplot as plt #plt.plot();
    import multiprocessing
    import time
    import pickle
    import os
    import datetime
#    from pylab import plot, ion, ioff, isinteractive, draw, clf, pause, savefig
#    def startNN(q,Eq):
#        print("Starting NN")
        #Set the parameters for the NN model
   
             
    
    ConcMean = np.load('ConcMeanDarkCtrl.npy')
    ConcStd = np.load('ConcStdDarkCtrl.npy')
    ConcStd[40] = 1 #Set Std for controlfishparameter to 1 
    CtrlMean = np.load('CtrlMeanDarkCtrl.npy')
    AllStd = np.load('AllStdDarkCtrl.npy')
    antiPsychList = np.load('AntipsychList.npy')
    plotConc = np.load('CheckModelPerformanceConc.npy')
    plotParam = np.load('CheckModelPerformanceParam.npy')
    plotParamN = 0.5*(np.tanh(0.1*(plotParam-CtrlMean)/AllStd)+1)
    plotConcN = 0.5*(np.tanh(0.1*(plotConc-ConcMean)/ConcStd)+1)
    basePath="C:\\Users\\Computer\\Documents\\ZebraFishComboAnalysis\\0__testing\\NNfig\\"
    
    paramSize = 50
    validSize = 5
    normParam = 1000.
    normConc = 100.
    batchSize = 50
    epochsNN = 6
    iteration = 0
    inputNum = 41
    outputNum = 1352
    aL=0.1
    numpy.random.seed(7)
    q = multiprocessing.Queue(maxsize=13)
    Eq = multiprocessing.Queue(maxsize=5)
    
    substancesCount = {}
    for i,v in enumerate(antiPsychList):
     concentrationCount = {}
     concentrationCount["0.1"]=0
     concentrationCount["0.3"]=0
     concentrationCount["1.0"]=0
     concentrationCount["3.0"]=0
     concentrationCount["10.0"]=0
     concentrationCount["25.0"]=0
     concentrationCount["33.0"]=0
     concentrationCount["50.0"]=0
     concentrationCount["100.0"]=0
     substancesCount[v] = concentrationCount
    
   
    
    def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
                os.makedirs(dir)
    
#    pNN = multiprocessing.Process(target=startNN, args=(q,Eq))
#    pNN.start()
    
    
    pE = multiprocessing.Process(target=fetchEva, args=(Eq,validSize))
    pE.start()
    pE2 = multiprocessing.Process(target=fetchEva, args=(Eq,validSize))
    pE2.start()
    pE3 = multiprocessing.Process(target=fetchEva, args=(Eq,validSize))
    pE3.start()
    time.sleep(0.1)
    pd = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd.start()
    time.sleep(0.1)
    pd2 = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd2.start()
    time.sleep(0.1)
    pd3 = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd3.start()
    time.sleep(0.1)
    pd4 = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd4.start()
    time.sleep(0.1)
    pd5 = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd5.start()
    time.sleep(0.1)
    pd6 = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd6.start()
    time.sleep(0.1)
    pd7 = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd7.start()
    time.sleep(0.1)
    pd8 = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd8.start()
    time.sleep(0.1)
    pd9 = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd9.start()
    time.sleep(0.1)
    pd10 = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd10.start()
    time.sleep(0.1)
    pd11 = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd11.start()
    time.sleep(0.1)
    pd12 = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd12.start()
    time.sleep(0.1)
    pd13 = multiprocessing.Process(target=fetch, args=(q,paramSize))
    pd13.start()
    time.sleep(0.1)
 
   
#    model = load_model('20juliFishAverageNormalized(2).h5')
    #Create the NN model
    inn = Input((inputNum,))
    
    r=math.sqrt(6/(30+100))
    lay=Dense(100,activation=None,kernel_initializer=initializers.RandomUniform(minval=-r, maxval=r))(inn)
    lay=LeakyReLU(alpha=aL)(lay)
    lay=Dropout(0.2)(lay)
    
    
    r=math.sqrt(6/(100+1000))
    lay=Dense(1000,activation=None,kernel_initializer=initializers.RandomUniform(minval=-r, maxval=r))(lay)
    lay=LeakyReLU(alpha=aL)(lay)
    lay=Dropout(0.2)(lay)
    
    r=math.sqrt(6/(1000+1000))
    lay=Dense(2000,activation=None,kernel_initializer=initializers.RandomUniform(minval=-r, maxval=r))(lay)
    lay=LeakyReLU(alpha=aL)(lay)
    lay=Dropout(0.2)(lay)
    
    r=math.sqrt(6/(1000+1000))
    lay=Dense(2000,activation=None,kernel_initializer=initializers.RandomUniform(minval=-r, maxval=r))(lay)
    lay=LeakyReLU(alpha=aL)(lay)
    layFinalDense=Dropout(0.2)(lay)
    
    lay=Reshape((2000,1), input_shape=(2000,))(layFinalDense)
    layFinalConv=Conv1D(filters=3,kernel_size=3,strides=3)(lay)
    
    layFlatten=Flatten()(layFinalConv)
    
    listStructures = []
    for i in range(5):
        layP = Dense(100,activation=None)(layFlatten)
        layP=LeakyReLU(alpha=aL)(layP)
        listStructures.append(layP)
    
    layStruct = Concatenate()(listStructures)
    
    listSameParam = []
    for i in range(52):
        layP = Dense(100,activation=None)(layStruct)
        layP=LeakyReLU(alpha=aL)(layP)
        layP = Dense(100,activation=None)(layP)
        layP=LeakyReLU(alpha=aL)(layP)
        listSameParam.append(layP)
    
    listL = []
    connectInd = 0
    for i in range(104):
        if(connectInd==52):connectInd = 0
        layP=Dense(128,activation=None)(listSameParam[connectInd])
        connectInd+=1
        layP=LeakyReLU(alpha=aL)(layP)
        layP=Dense(60,activation=None)(layP)
        layP=LeakyReLU(alpha=aL)(layP)
        layP=Dense(13,activation=None)(layP)
        layP=LeakyReLU(alpha=aL)(layP)
        
        listL.append(layP)
        
    
    lay=Concatenate()(listL)
    
   
        
    
    lay=Dense(outputNum,activation=None)(lay)
    lay=LeakyReLU(alpha=aL)(lay)
    model=Model(inputs=inn,outputs=lay)
    model.summary()
    adam = Adam(lr=0.001)
    #Compile the created model
    model.compile(loss='logcosh', optimizer=adam, metrics=['accuracy'])
    


    
    
    for index in range(0,10000000):
        #Repeat the training 10000 times
        now = datetime.datetime.now()
        print("*****************Creating graph folder************")
        FolderName=str(now.year)+'-'+str(now.month)+'-'+str(now.day)+"\\"
        assure_path_exists(basePath+FolderName)
        
        
        iteration += 1
        print("*****************Fetching data********************")
        #IT GETS STUCK HERE AFTER TWO ROUNDS!
        parameterNp, concentrationNp = q.get()
        parameterNpEva, concentrationNpEva = Eq.get()
        
        countIndividuals(concentrationNp,substancesCount,antiPsychList)
        #Normalize parameters
        parameterNp = 0.5*(np.tanh(0.1*(parameterNp-CtrlMean)/AllStd)+1)
        concentrationNp = 0.5*(np.tanh(0.1*(concentrationNp-ConcMean)/ConcStd)+1)
#        parameterNp = (parameterNp-CtrlMean)/AllStd
#        concentrationNp = (concentrationNp-ConcMean)/ConcStd        
        concLabelVector = concentrationNpEva    
        parameterNpEva = 0.5*(np.tanh(0.1*(parameterNpEva-CtrlMean)/AllStd)+1)
        concentrationNpEva = 0.5*(np.tanh(0.1*(concentrationNpEva-ConcMean)/ConcStd)+1)
#        parameterNpEva = (parameterNpEva-CtrlMean)/AllStd
#        concentrationNpEva = (concentrationNpEva-ConcMean)/ConcStd        
        
#        parameterNpEva=parameterNpEva/normParam
#        concentrationNpEva=concentrationNpEva/normConc
#        
#        parameterNp=parameterNp/normParam
#        concentrationNp=concentrationNp/normConc
        model.fit(concentrationNp, parameterNp, epochs=epochsNN, batch_size=batchSize)
        
        

        
        # evaluate the model
        scores = model.evaluate(concentrationNpEva, parameterNpEva)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        print("This was round "+str(index))
        
        file = open(str(now.month)+'-'+str(now.day)+"-NNLog.txt","a") 
        file.write("Round number "+str(index)+" ")
        file.write("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        file.write("\n")  
        file.close() 
        
        
      
        plt.ion()
        datax=model.predict(concentrationNpEva).reshape([-1])
        datay=parameterNpEva.reshape([-1])
        
        
        plt.pause(0.01)
        
        fig1 = plt.figure(1)
        plt.clf()
        for j in range(validSize):
            plt.axhspan(np.min(datay[np.nonzero(datay)]), max(datay),xmin=j*1/validSize, xmax=(j+1)*1/validSize, facecolor='y', alpha=0.5*(1+(-1)**(j+1)))
        fig1.suptitle('Random Individuals')
        
        plt.plot(datax,label='Prediction')
        plt.pause(0.01)
        
        plt.plot(datay,alpha=0.5,label='Correct')
        
        plt.pause(0.01)
        #
        #plt.title('Random Individuals')
#        plt.xlabel('Parameters')
        
        
        plt.xticks([(j*1352)+676 for j in range(validSize)], ['ind '+str(j) for j in range(validSize)], rotation='vertical')
        plt.ylabel('Parameter value')
        plt.legend(loc='best')
        plt.show()
        print(concLabelVector)
        #plt.draw()
        plt.pause(0.01)
        if(iteration==100):
            RandomIndFigFolder="RandomIndividuals\\"
            assure_path_exists(basePath+FolderName+RandomIndFigFolder)
            plt.savefig(basePath+FolderName+RandomIndFigFolder+'plot'+str(index)+'.png')
        
        for i,v in enumerate(plotConc):
        
            datax=model.predict(plotConcN[i].reshape([1,41])).reshape(-1)
            datay=plotParamN[i]
            
            #plt.clf()
            plt.pause(0.01)
            
            fig2 = plt.figure(i+2)
            plt.clf()
            plt.axhspan(np.min(datay[np.nonzero(datay)]), max(plotParamN[i]), xmax=0.5, facecolor='g', alpha=0.5)
            titleString = ''
            for ind,val in enumerate(plotConc[i][:-1]):
                if(val!=0.0):
                    titleString=titleString+antiPsychList[ind]+' '+str(val)+' '
            if(titleString == ''):
                titleString = 'Control fish'
            fig2.suptitle(titleString)
            #plt.title('Random Individuals')
            plt.plot(datax,label='Prediction',color='red')
            plt.pause(0.01)
            
            
            plt.plot(datay,alpha=0.5,label='Correct',color='blue')
            plt.pause(0.01)
            
            plt.text(0, min(plotParamN[i]), 'Lightdark')
            plt.text(676, min(plotParamN[i]), '|')
            plt.text(1300, min(plotParamN[i]), 'PTZ')
            plt.xlabel('Parameters')
            plt.ylabel('Parameter value')
            plt.legend(bbox_to_anchor=(0, 0.85), loc=3, borderaxespad=0.)
            plt.show()
            #plt.draw()
            plt.pause(0.01)
            
            if(iteration==100):
                SpecificConcFolder = str(i)+"\\"
                assure_path_exists(basePath+FolderName+SpecificConcFolder)
                plt.savefig(basePath+FolderName+SpecificConcFolder+titleString+'-'+str(index)+'.png')
                
        countList=[0.0 for i in range(len(substancesCount))]
        for i,v in enumerate(substancesCount):
            for ind,va in enumerate(substancesCount[v]):
                countList[i] += float(substancesCount[v][va])
        fig10 = plt.figure(10)
        plt.clf()
        index = np.arange(len(antiPsychList))
        bar_width=0.35
        plt.bar(index, countList, bar_width)
        plt.xticks([j for j in range(len(substancesCount))], antiPsychList, rotation='vertical')
        if(iteration==100):
                indCountFolder="IndividualCount\\"
                assure_path_exists(basePath+FolderName+indCountFolder)
                plt.savefig(basePath+FolderName+indCountFolder+'NumberOfIndividuals.png')
                
            
#            draw()
           
        if(iteration==100):
#            model.save("C:\\Users\\Computer\\Documents\\ZebraFishComboAnalysis\\0__testing\\27JulyParallellParam.h5")
            
            #Save architecture
            architecture = model.to_json()
            with open("modelArchitecture"+str(now.month)+"-"+str(now.day)+".json", "w") as json_file:             
                json_file.write(architecture) 
            #Save weights
            weights=model.get_weights()
            model.save_weights("weights"+str(now.month)+"-"+str(now.day)+".h5")
#            with open("weight27July.pkl", 'wb') as f:
#                pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)
            #Save optimizer state
            symbolic_weights = getattr(model.optimizer, 'weights')
            weight_values = K.batch_get_value(symbolic_weights)
            with open("optimizer"+str(now.month)+"-"+str(now.day)+".pkl", 'wb') as f:
                pickle.dump(weight_values, f)
            
            
            
#            optimizerState = model.optimizer.get_state()
#            with open("optimizerState27July.pkl", 'wb') as f:
#                pickle.dump(optimizerState, f, pickle.HIGHEST_PROTOCOL)
            
            iteration = 0
                
            
            
