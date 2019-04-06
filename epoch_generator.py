'''
Random dataset generator for every epoch that ensures equal no of training samples
superclass: Keras.utils.Sequence

'''
import numpy as np
import keras
import cv2

class generator_epoch(keras.utils.Sequence):
    def __init__(self,X,Y, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        
        self.dims = X[0].shape

        self.shuffle = shuffle

        #creates classes
        self.classes=list(np.unique(Y))
        self.classes.sort()

        self.class_no=len(self.classes)
        self.class_set=[[] for i in range(self.class_no)]
        for i,j in zip(X,Y):
            self.class_set[self.classes.index(j)].append([i,j])

        self.class_len=[len(lst) for lst in self.class_set]
        self.class_small=min(self.class_len)
        self.class_bound=np.array([[0,self.class_small],]*self.class_no)

        self.epoch_list=[]
        for _id,lst in enumerate(self.class_set):
            a,b=self.class_bound[_id]
            self.epoch_list.extend(lst[a:b])
        np.random.shuffle(self.epoch_list)
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.epoch_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        data_points = self.epoch_list[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(data_points)
        return(X,y)

    def on_epoch_end(self):

        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        #update each epoch set
        self.epoch_list=[]
        for _id,lst in enumerate(self.class_set):
            a,b=self.class_bound[_id]
            if b+self.class_small<class_len[_id]:
                a=b
                b+=self.class_small
                self.epoch_list.extend(lst[a:b])                
            elif b+self.class_small>class_len[_id]:
                a=b
                b=self.class_small-(self.class_len[_id]-b)
                self.epoch_list.extend((lst[a:]+lst[:b]))
            else:
                self.epoch_list.extend(lst[a:b])
            self.class_bound[_id]=[a,b]    
        np.random.shuffle(self.epoch_list)



    def __data_generation(self, data_points):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty(((self.batch_size,)+self.dims))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i,data in enumerate(data_points):
            # Store sample
            # im=cv2.imread(ID)
            X[i,] = data[0].astype(np.float32)
            # Store class
            y[i] = data[1]

        return(X, keras.utils.to_categorical(y, num_classes=self.class_no))


