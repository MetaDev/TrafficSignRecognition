from sklearn.ensemble import RandomForestClassifier
import numpy 
import scipy 
import data_loading as loader
from skimage import exposure, color
import score_calculation
import sklearn.cross_validation as cv
print("Loading images")
#images, classes = loader.loadProblematicImagesAndClasses()
images, classes = loader.loadTrainingAndClasses()

amount = len(images)


print("Making thumbnails")

size=50
thumbs = [scipy.misc.imresize(img, (size, size)) for img in images]

thumbs=numpy.array(thumbs)


#normalize image

#grayscale
imageGray = color.rgb2gray(image)
#equlize histogram
imageGray=exposure.equalize_hist(imageGray)

#create the training & test sets, skipping the header row with [1:]
   
target = [x[0] for x in dataset]
rain = [x[1:] for x in dataset]

    
#create and train the random forest
#multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target)


#print("Producing KFold indexes")
#kfold = cv.KFold(amount, n_folds = 5, shuffle = True)
#
#print("K-fold prediction score")
#score = cross_validation.cross_val_score(nn, thumbs, int_classes, cv=3)
#print(score)
#print(numpy.mean(score),numpy.std(score))
#
#print("K-fold log loss prediction score")
#scores = score_calculation.loglossKFold(thumbs,int_classes,nn,8)
#print(scores)
#print(numpy.mean(scores),' ' ,numpy.std(scores))

