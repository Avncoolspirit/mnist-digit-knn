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
        number_pixel_dict[dataset[index_label][0]]= {'count':1,'number_array':dataset[index_label][1:],'index':index_label}
    else:
        number_pixel_dict[dataset[index_label][0]]['count']+=1
        
        
total_prob=0
for label in range(0,10):
    pixel_array = np.array(number_pixel_dict[label]['number_array'], dtype = 'uint8').reshape((28,28))
    prior_prob = (number_pixel_dict[label]['count'])/42000.0
    number_pixel_dict[label]['prior_prob']=prior_prob  
#    plt.title('Label is {Label}'.format(Label = labels))
#    plt.imshow(pixel_array, cmap = 'gray')
#    plt.show()

for label in range(0,10):
    print("Prior Probability for" , label, "is ", number_pixel_dict[label]['prior_prob'])
    
tile_matrix = np.tile(number_pixel_dict[1]['number_array'],(42000,1))
dataset_numpy = pixels.to_numpy()
subtraction_matrice = np.subtract(tile_matrix,dataset_numpy)
ones=np.ones(784)
square_row=np.square(subtraction_matrice) 
distance_squared =  square_row*ones
distance = sorted(np.sum(distance_squared,axis=1))

    

