
#%% Import required libraries 
import numpy
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import scipy
from skimage import morphology
from keras.models import Model, load_model
from keras.layers.merge import concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.normalization import BatchNormalization
import nibabel as nib
import glob
from matplotlib import pyplot as plt


#%% Load 3D LGE CMRIs along with the myocardium masks

path1 = r'Please enter the path where the LGE CMRIs in .nii format are located.'
LGEs = glob.glob(path1 + "/*")

path2 = r'Please enter the path where the myocardium masks in .nii format are located.'
MYOs = glob.glob(path2 + "/*")

#%% Prepare training samples

x_unet = 256
y_unet = 256

data_train = numpy.zeros((1,x_unet*y_unet))
mask_train =  numpy.zeros((1,x_unet*y_unet))

for n in range(18): 
    
    data_lge = nib.load(LGEs[n]);
    lge = data_lge.get_data()
    x,y,z = lge.shape
    
    lge_norm = numpy.zeros((x,y,z))
    for slice_no in range (y):
        lge_slice = lge[:,slice_no,:]
        for a in range (x):
            for b in range (z):
                if lge_slice[a,b] > 1000:
                    lge_slice[a,b] = numpy.median(lge_slice)
        lge_slice = (lge_slice-lge_slice.min())/(lge_slice.max()-lge_slice.min())
        lge_norm[:,slice_no,:] = lge_slice
      
   
    
    data_myo = nib.load(MYOs[n]);
    myo = data_myo.get_data()
    

    data = numpy.zeros((1,x_unet*y_unet))
    mask = numpy.zeros((1,x_unet*y_unet))
    
    x_pad = int(x_unet - x)
    y_pad = int(y_unet - z)
    
    for page in range(0,y):    
        lge_slice = lge_norm[:,page,:]
        myo_slice = myo[:,page,:]
        
        if (numpy.max(myo_slice) != 0):
            lge_slice = numpy.pad(lge_slice, ((0, x_pad),(0, y_pad)), 'wrap')
            myo_slice = numpy.pad(myo_slice, ((0, x_pad),(0, y_pad)), 'wrap')
            
            
            lge_slice = lge_slice.reshape(1,(x_unet*y_unet))
            myo_slice = myo_slice.reshape(1, (x_unet*y_unet)) 
           
            data = numpy.vstack((data, lge_slice))
            mask = numpy.vstack((mask,myo_slice))
          
            

    data = numpy.delete(data, (0), axis=0)     
    data_train = numpy.vstack((data_train, data))   
    
    mask = numpy.delete(mask, (0), axis=0)     
    mask_train = numpy.vstack((mask_train, mask)) 
        
data_train = numpy.delete(data_train, (0), axis=0) 
mask_train = numpy.delete(mask_train, (0), axis=0) 

#% Reshape training dataset
data_train = data_train.reshape(data_train.shape[0], x_unet, y_unet, 1)
mask_train = mask_train.reshape(mask_train.shape[0], x_unet, y_unet, 1)


#%% U-Net Architecture
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


filter_no = 32

inputs = Input((x_unet, y_unet, 1))


conv1 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv1)
conv1 = BatchNormalization()(conv1)


pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
conv2 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv2)
conv2 = BatchNormalization()(conv2)


pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
conv3 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv3)
conv3 = BatchNormalization()(conv3)


pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
conv4 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = Dropout(0.5)(conv4)

pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
conv5 = Conv2D(filter_no*16, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool4)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(filter_no*16, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv5)
conv5 = BatchNormalization()(conv5)


up1 = UpSampling2D(size = (2,2))(conv5)
merge1 = concatenate([conv4,up1], axis = 3)
conv6 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge1)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv6)
conv6 = BatchNormalization()(conv6)


up2 = UpSampling2D(size = (2,2))(conv6)
merge2 = concatenate([conv3,up2], axis = 3)
conv7 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge2)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv7)
conv7 = BatchNormalization()(conv7)


up3 = UpSampling2D(size = (2,2))(conv7)
merge3 = concatenate([conv2,up3], axis = 3)
conv8 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge3)
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv8)
conv8 = BatchNormalization()(conv8)


up4 = UpSampling2D(size = (2,2))(conv8)
merge4 = concatenate([conv1,up4], axis = 3)
conv9 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge4)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv9)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(2, 3, strides=(1, 1), activation = 'relu', padding = 'same')(conv9)
conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = Model(input = inputs, output = conv9)

model.compile(optimizer='adadelta', loss=dice_coef_loss, metrics=[dice_coef])	
model.summary()


#%% Train Model and save the best trained one
fname= "segment_myo_xz1.hdf5"
checkpointer = ModelCheckpoint(fname, verbose=1, save_best_only=True)
results = model.fit(data_train, mask_train, validation_split=0.2, shuffle=True, batch_size=10, epochs=70, callbacks=[checkpointer])		

#%%
# summarize history for accuracy
plt.plot(results.history['dice_coef'])
plt.plot(results.history['val_dice_coef'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#%% Save trained model 

fname= "segment_myo_xz.hdf5"
model.save(fname, overwrite = True)



def model_evaluate(data,mask):
    dsc = []
    acc = []
    prec = []
    rec = []

    for k in range(len(data)):
        mask_sample = mask[k,:,:,:]
        mask_sample = mask_sample.reshape(x_unet, y_unet)
        img_test = data[k,:, :, :]
        img_test = img_test.reshape(1, x_unet, y_unet, 1)
        img_pred = model.predict(img_test, batch_size=1, verbose=1)
        img_pred = img_pred.reshape(x_unet, y_unet)
        img_pred  = (img_pred  > 0.5).astype(numpy.uint8)
        
        seg_clean = numpy.array(img_pred, bool)
        seg_clean = morphology.remove_small_objects(seg_clean,100) 
        seg_clean = seg_clean*1 
        
        seg_clean = scipy.ndimage.morphology.binary_dilation(seg_clean, iterations=3)
        seg_clean = scipy.ndimage.morphology.binary_erosion(seg_clean) 
        seg_clean = seg_clean*1    
        seg_clean = seg_clean[:x, :z]
        
        mask_clean = scipy.ndimage.morphology.binary_dilation(mask_sample,  iterations=2)
        mask_clean = scipy.ndimage.morphology.binary_erosion(mask_clean)
        mask_clean = mask_clean*1
        mask_clean = mask_clean[:x, :z]
        
        img_test = img_test.reshape(x_unet, y_unet)
        img_orig[:,k,:] = img_test[:x, :z]
        myo_m[:,k,:] = mask_clean
        myo_a[:,k,:] = seg_clean
       
                
        y_true = numpy.reshape(mask_clean, (x*z,1))
        y_pred = numpy.reshape(seg_clean, (x*z,1))
        
        dsc = numpy.append(dsc,f1_score(y_true, y_pred, average='macro'))
        acc = numpy.append(acc,accuracy_score(y_true, y_pred))
        prec = numpy.append(prec,precision_score(y_true, y_pred, average='macro'))
        rec = numpy.append(rec,recall_score(y_true, y_pred, average='macro'))
    
    dsc = round(numpy.mean(dsc)*100,2)
    acc = round(numpy.mean(acc)*100,2)
    prec = round(numpy.mean(prec)*100,2)
    rec = round(numpy.mean(rec)*100,2)
    return(dsc,acc,prec,rec)


#%% Create test dataset and test unseen images

x_unet = 256
y_unet = 256

dice_index =[]
accuracy = []
precision = []
recall = []

slice_total = 0

for n in range(18,34): 
    
    data_lge = nib.load(LGEs[n]);
    lge = data_lge.get_data()
    x,y,z = lge.shape
    
    img_orig = numpy.zeros((x,y,z))
    myo_m = numpy.zeros((x,y,z))
    myo_a = numpy.zeros((x,y,z))
    
    lge_norm = numpy.zeros((x,y,z))
    for slice_no in range (y):
        lge_slice = lge[:,slice_no,:]
        for a in range (x):
            for b in range (z):
                if lge_slice[a,b] > 1000:
                    lge_slice[a,b] = numpy.median(lge_slice)
        lge_slice = (lge_slice-lge_slice.min())/(lge_slice.max()-lge_slice.min())
        lge_norm[:,slice_no,:] = lge_slice
     
    
    data_myo = nib.load(MYOs[n]);
    myo = data_myo.get_data()    
 
    data = numpy.zeros((1,x_unet*y_unet))
    mask = numpy.zeros((1,x_unet*y_unet))
    
    x_pad = int(x_unet - x)
    y_pad = int(y_unet - z)
    
    for page in range(0,y):    
        lge_slice = lge_norm[:,page,:]
        myo_slice = myo[:,page,:]
        
        
        lge_slice = numpy.pad(lge_slice, ((0, x_pad),(0, y_pad)), 'wrap')
        myo_slice = numpy.pad(myo_slice, ((0, x_pad),(0, y_pad)), 'wrap')
        
        lge_slice = lge_slice.reshape(1,(x_unet*y_unet))
        myo_slice = myo_slice.reshape(1, (x_unet*y_unet)) 
        
        data = numpy.vstack((data,lge_slice ))
        mask = numpy.vstack((mask,myo_slice))

    data = numpy.delete(data, (0), axis=0)     
    
    mask = numpy.delete(mask, (0), axis=0)     
    
    data = data.reshape(data.shape[0], x_unet, y_unet, 1)
    mask = mask.reshape(mask.shape[0], x_unet, y_unet, 1)
        
    p1,p2,p3,p4 = model_evaluate(data,mask)
    dice_index = numpy.append(dice_index,p1)
    accuracy = numpy.append(accuracy,p2)
    precision = numpy.append(precision,p3)
    recall = numpy.append(recall,p4)
    slice_total += data.shape[0]

     
print('Mean Values:')    
print('DI is :', round(numpy.mean(dice_index),2) , '+', round(numpy.std(dice_index),2))
print('Acc. is :', round(numpy.mean(accuracy),2), '+', round(numpy.std(accuracy),2))
print('Precision is :', round(numpy.mean(precision),2), '+', round(numpy.std(precision),2))
print('Recall is :', round(numpy.mean(recall),2), '+', round(numpy.std(recall),2))

print('Median Values:') 
print('DI is :', round(numpy.median(dice_index),2) , '+', round(numpy.std(dice_index),2))
print('Acc. is :', round(numpy.median(accuracy),2), '+', round(numpy.std(accuracy),2))
print('Precision is :', round(numpy.median(precision),2), '+', round(numpy.std(precision),2))
print('Recall is :', round(numpy.median(recall),2), '+', round(numpy.std(recall),2))

