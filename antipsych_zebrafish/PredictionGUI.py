from tkinter import *
import numpy as np

antiPsychList = np.load('AntipsychList.npy')
root= Tk()
entryList = []
inputVector = []

def predict():
    for i,v in enumerate(entryList):
        inputVector.append(float(v.get()))
    inputVector.append(1.0)
    print(inputVector)


label_1 = Label(root, text="...Zebrafish crystal ball...")
label_1.grid(columnspan=2)
for i,v in enumerate(antiPsychList):
    
    label_1 = Label(root, text=v)
    entry_1 = Entry(root)
    label_1.grid(row=i+1, sticky=E)
    entry_1.grid(row=i+1, column=1)
    
    entry_1.insert(0,"0.0")
    entryList.append(entry_1)

button_1 = Button(root, text="Predict", command=predict)
button_1.grid(row=len(antiPsychList)+1, column=1)
root.mainloop()

