import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

number_pixel_dict = {}
pixels = df.iloc[ : , 1:]
labels = df.iloc[ : , 0]
dataset = df.values
pixels_0_1 = df.loc[df['label'].isin([0,1])]
pixels_0_1_px = pixels_0_1.iloc[ : , 1:]
pixels_0_1_size = len(pixels_0_1.index)
pixels_0_1_labels = pixels_0_1.iloc[ : , 0]

def calculate_l2_norm(tile_matrix, dataset_numpy):
    subtraction_matrix = np.subtract(tile_matrix,dataset_numpy)
    ones=np.ones(784)
    square_row=np.square(subtraction_matrix) 
    transpose_one = np.transpose([ones])
    distance_squared = (np.dot(square_row, transpose_one))
    
    euclidean_distances = pd.DataFrame(distance_squared)
    euclidean_distances = euclidean_distances[euclidean_distances > .01].min(axis=1)
    min_index = euclidean_distances.idxmin(axis=0, skipna=True)
    return min_index

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

for label in range(0,10):
    tile_matrix = np.tile(number_pixel_dict[label]['number_array'],(42000,1))
    dataset_numpy = pixels.to_numpy()
    min_index = calculate_l2_norm(tile_matrix, dataset_numpy)
    print(min_index)

for index in range(0, len(pixels_0_1_labels)):
    pixels_0_1_labels.sort_index()
    print(index)
    tile_matrix = np.tile(pixels_0_1_px.iloc[ index , : ],(pixels_0_1_size,1))
    min_index = calculate_l2_norm(tile_matrix, pixels_0_1_px)
    print("Nearest neighbour for index: " , index, "is ", pixels_0_1_labels.iloc[min_index])
        
    

    



    

