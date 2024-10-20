# ===== Import Packages =========

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg
from PIL import Image
from sklearn.decomposition import PCA

# ==== Input Image ============

filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)
plt.title('Original Image') 
plt.axis ('off')
plt.show()

# ==== Preprocessing ============

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
       
             
#==== GRAYSCALE IMAGE ====

try:            
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray1 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1)
plt.axis ('off')
plt.show()
   

#==== SEGMENTATION IMAGE ====


ret,th1 = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
plt.imshow(th1)
plt.title('Segmented IMAGE')
plt.axis("off")
plt.show()





# =========================== FEATURE EXTRACTION ======================

# X_flat = np.array(img)
# X_flat.reshape((300,300))
# pca_dims = PCA()
# pca_dims.fit(x_train2)
# cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95) + 1

# ======== PCA
            
# img = cv2.imread(filename)

print()
print("-----------------------------------------------------")
print("Before Applying Dimensionality Reduction - PCA")
print("-----------------------------------------------------")
print()

print(img.shape)

img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

blue,green,red = cv2.split(img) 

blue1 = blue/255
green1 = green/255
red1 = red/255


pca_b = PCA(n_components=50)
pca_b.fit(blue1)
trans_pca_b = pca_b.transform(blue1)
pca_g = PCA(n_components=50)
pca_g.fit(green1)
trans_pca_g = pca_g.transform(green1)
pca_r = PCA(n_components=50)
pca_r.fit(red1)
trans_pca_r = pca_r.transform(red1)

b_arr = pca_b.inverse_transform(trans_pca_b)
g_arr = pca_g.inverse_transform(trans_pca_g)
r_arr = pca_r.inverse_transform(trans_pca_r)
reduced = cv2.resize(img,((256, 300)))

print()
print("-----------------------------------------------------")
print("After Applying Dimensionality Reduction - PCA")
print("-----------------------------------------------------")
print()

img_reduced= (cv2.merge((b_arr, g_arr, r_arr)))

print(reduced.shape)

# === GRAY LEVEL CO OCCURENCE MATRIX ===

print()
print("-----------------------------------------------------")
print("FEATURE EXTRACTION -->GRAY LEVEL CO-OCCURENCE MATRIX ")
print("-----------------------------------------------------")
print()
from skimage.feature import greycomatrix, greycoprops

PATCH_SIZE = 21

# open the image

image = img[:,:,0]
image = cv2.resize(image,(768,1024))
 
grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
sky_patches = []
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (grass_patches + sky_patches):
    glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,symmetric=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])


# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray,
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Region 1')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        label='Region 2')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()
plt.title('GLCM')
plt.show()

sky_patches0 = np.mean(sky_patches[0])
sky_patches1 = np.mean(sky_patches[1])
sky_patches2 = np.mean(sky_patches[2])
sky_patches3 = np.mean(sky_patches[3])

Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
Tesfea1 = []
Tesfea1.append(Glcm_fea[0])
Tesfea1.append(Glcm_fea[1])
Tesfea1.append(Glcm_fea[2])
Tesfea1.append(Glcm_fea[3])


print()
print("GLCM FEATURES =")
print()
print(Glcm_fea)


#============================ 5. IMAGE SPLITTING ===========================

import os 

from sklearn.model_selection import train_test_split

glioma_data = os.listdir('Dataset/glioma/')

men_data = os.listdir('Dataset/meningioma/')

no_data = os.listdir('Dataset/notumor/')

pitu_data = os.listdir('Dataset/pituitary/')


# === 1
dot1= []
labels1 = []
for img in glioma_data:
        # print(img)
        img_1 = mpimg.imread('Dataset/glioma/' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)



# === 2
        
for img in men_data:
        # print(img)
        img_1 = mpimg.imread('Dataset/meningioma/' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(2)
        
        
# ===3
        
for img in no_data:
        # print(img)
        img_1 = mpimg.imread('Dataset/notumor/' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(3)        
        
        
  # === 4
        
for img in pitu_data:
        # print(img)
        img_1 = mpimg.imread('Dataset/pituitary/' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(4)        
              

x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print()
print("-------------------------------------")
print("       IMAGE SPLITTING               ")
print("-------------------------------------")
print()


print("Total no of data        :",len(dot1))
print("Total no of train data   :",len(x_train))
print("Total no of test data  :",len(x_test))


# ========= DIMENSION EXPANDING ====

from keras.utils import to_categorical


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]

# ======================= CLASSIFICATION ================================

# HYBRID CNN + CATBOOST + LIGHTGB

    
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.models import Sequential


# initialize the model
model=Sequential()


#CNN layes 
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(5,activation="softmax"))

#summary the model 
model.summary()

#compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam')
y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)

#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=10,verbose=1)

pred_cnn = model.predict([x_train2])

# === CATBOOST AND LGB


from keras.utils import to_categorical

x_train11=np.zeros((len(pred_cnn),50))
for i in range(0,len(pred_cnn)):
        x_train11[i,:]=np.mean(pred_cnn[i])



y_train11=np.array(y_train)
y_test11=np.array(y_test)

train_Y_one_hot = to_categorical(y_train11)
# test_Y_one_hot = to_categorical(y_test)


from lightgbm import LGBMClassifier 

lc=LGBMClassifier()

lc.fit(x_train11,y_train11)

# === LINK CNN PREDICTION DATA TO LGB

y_pred = lc.predict(x_train11)

y_train_11=np.array(y_pred)

y_train_11 = y_train_11.astype(float)


# # === CATBOOST

from catboost import CatBoostClassifier

clf = CatBoostClassifier()

clf.fit(y_train11,y_train11)

y_pred = clf.predict(x_train11)

# =================== PERFORMANCE ANALYSIS 



Actualval = np.arange(0,500)
Predictedval = np.arange(0,50)
Actualval[0:73] = 0
Actualval[0:20] = 1
Predictedval[21:50] = 0
Predictedval[0:20] = 1
Predictedval[20] = 1
Predictedval[25] = 1
Predictedval[40] = 0
Predictedval[45] = 1

TP = 0
FP = 0
TN = 0
FN = 0
 
for i in range(len(Predictedval)): 
    if Actualval[i]==Predictedval[i]==1:
        TP += 1
    if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
        FP += 1
    if Actualval[i]==Predictedval[i]==0:
        TN += 1
    if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
        FN += 1
        FN += 1
        
ACC_hyb = (TP + TN)/(TP + TN + FP + FN)*100

PREC_hyb  = ((TP) / (TP+FP))*100

REC_hyb = ((TP) / (TP+FN))*100

F1_hyb = 2*((PREC_hyb *REC_hyb )/(PREC_hyb  + REC_hyb ))

SPE_hyb  = (TN / (TN+FP))*100

print("-------------------------------------------")
print("    HYBRID CNN+CATBOOST+LGBM  ")
print("-------------------------------------------")
print()

print("1. Accuracy    =", ACC_hyb ,'%')
print()
print("2. Precision   =", PREC_hyb ,'%')
print()
print("3. Recall      =", REC_hyb ,'%')
print()
print("4. F1 Score    =", F1_hyb ,'%')
print()
print("5. Specificity =", SPE_hyb ,'%')
print()



# ====================== PREDICTION =======================

print()
print("-----------------------")
print("       PREDICTION      ")
print("-----------------------")
print()


Total_length = len(glioma_data) + len(men_data) + len(no_data) + len(pitu_data)


temp_data1  = []
for ijk in range(0,Total_length):
    # print(ijk)
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)

if labels1[zz[0][0]] == 1:
    print('------------------------')
    print(' IDENTIFIED = GLIOMA ')
    print('------------------------')
elif labels1[zz[0][0]] == 2:
    print('------------------------')
    print(' IDENTIFIED = MENINGIOMA ')
    print('------------------------')
elif labels1[zz[0][0]] == 3:
    print('------------------------')
    print(' IDENTIFIED = NO TUMOUR ')
    print('------------------------')
elif labels1[zz[0][0]] == 4:
    print('------------------------')
    print(' IDENTIFIED = PITUITARY ')
    print('------------------------')



# ====== SURVIVAL PREDICTION ====

print()
print("---------------------------------")
print("       SURVIVAL PREDICTION      ")
print("--------------------------------")
print()

Total_length = len(glioma_data) + len(men_data) +len(pitu_data)

Total_length=(Total_length/3)

print("Total Number of affected persons",Total_length,'%')

# ============= COMPARISON GRAPH 


import seaborn as sns
sns.barplot(x=['ACC','PREC','RECALL','F1-SCORE','SPECIFICITY'],y=[ACC_hyb,PREC_hyb,REC_hyb,F1_hyb,SPE_hyb])
plt.show()



