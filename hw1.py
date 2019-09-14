import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

number_pixel_dict = {}
pixels = df.iloc[ : , 1:]
labels = df.iloc[ : , 0]
dataset = df.values

for index_label in range(0,len(dataset)):
    if dataset[index_label][0] not in number_pixel_dict.keys():
        number_pixel_dict[dataset[index_label][0]]= {'count':1,'number_array':dataset[index_label][1:]}
    else:
        number_pixel_dict[dataset[index_label][0]]['count']+=1
        
        
total_prob=0
for label in range(0,10):
    pixel_array = np.array(number_pixel_dict[label]['number_array'], dtype = 'uint8').reshape((28,28))
    #plt.title('Label is {Label}'.format(Label = labels))
    plt.imshow(pixel_array, cmap = 'gray')
    plt.show()

for label in range(0,10):
    prior_prob = (number_pixel_dict[label]['count'])/42000.0
    print("Prior Probability for" , label, "is ", prior_prob)


