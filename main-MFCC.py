import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import itertools

ROOT_PATH = os.getcwd ()
DATA_PATH = ROOT_PATH + '/data'

mapDatabyName = {}
mapEmotionbyName = {}
label_age = []
label_emotion = []

EPOCH = 2
BATCH_SIZE = 5000
LEARNING_RATE = 0.0001


rawdata = pd.read_csv('MFCC/MFCC_Emotion.csv', delimiter=',')

X = rawdata.iloc[:, :-1]
label = rawdata.iloc[:, -1]

def scale_data (x):
    scaler = StandardScaler ().fit (x)
    # scale everything
    scaled = scaler.transform (x)
    return scaled


def getEncodedLabel (inputList):
    encoder = LabelEncoder ()
    encoder.fit (inputList)
    classList = encoder.classes_
    encodedList = encoder.transform (inputList)
    return encodedList, classList


encoder = LabelEncoder ()
encoder.fit (label)
classList = np.unique(label)
num_emotion = len (classList)

print('num emotion: {}'.format(len (classList)))
print('emotions: {}'.format(classList))

y = encoder.transform (label)

# train/test split
X_train, X_test, Y_train, Y_test = train_test_split (X, y, test_size=0.2, shuffle=True, random_state=123)
# one-hot encoding labels

Y_train = tf.keras.utils.to_categorical (Y_train, num_emotion)
Y_test = tf.keras.utils.to_categorical (Y_test, num_emotion)

# reshape to (~, 43, 1)
X_train = X_train.values.reshape(-1, 43, 1)
X_test = X_test.values.reshape(-1, 43, 1)

def buildNet_CNN (inputs, reuse=True, trainable=True):
    # transpose = tf.keras.layers.Permute ((2, 1)) (inputs)

    # lstm1 = tf.keras.layers.GRU (518, return_sequences=True, kernel_initializer='glorot_uniform',
    #                              activation='relu',
    #                              dropout=0.2) (transpose)
    # lstm2 = tf.keras.layers.GRU (256, return_sequences=True, kernel_initializer='glorot_uniform',
    #                              activation='relu',
    #                              dropout=0.2) (lstm1)

    conv1 = tf.keras.layers.Conv1D (filters=128, kernel_size=5,
                                    strides=2, padding='same',
                                    activation='relu',
                                    name='conv1') (inputs)
    bn1 = tf.keras.layers.BatchNormalization (name='bn1') (conv1)
    drop1 = tf.keras.layers.Dropout (rate=0.4, name='drop1') (bn1)

    conv2 = tf.keras.layers.Conv1D (filters=256, kernel_size=5,
                                    strides=2, padding='same',
                                    activation='relu',
                                    name='conv2') (drop1)
    bn2 = tf.keras.layers.BatchNormalization (name='bn2') (conv2)
    drop2 = tf.keras.layers.Dropout (rate=0.4, name='drop2') (bn2)

    flat = tf.keras.layers.Flatten (name='flatten') (drop2)
    fc = tf.keras.layers.Dense (units=128, activation='relu', name='fc2') (flat)
    drop3 = tf.keras.layers.Dropout (rate=0.4, name='drop3') (fc)
    output = tf.keras.layers.Dense (units=num_emotion, activation='softmax') (drop3)
    return output


tensor_X = tf.keras.layers.Input (shape=[X_train.shape[1], 1])
tensor_pred = buildNet_CNN (inputs=tensor_X)

model = tf.keras.Model (inputs=tensor_X, outputs=tensor_pred)
model.summary ()

adam = tf.keras.optimizers.Adam (lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile (optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# tensor_board = tf.keras.callbacks.TensorBoard (
#     log_dir='log/' + 'logfile',
#     histogram_freq=0,
#     write_graph=False, write_images=True,
#     embeddings_freq=0)
csv_logger = tf.keras.callbacks.CSVLogger('./logs/history_MFCC.csv', append=True)

history = model.fit (X_train, Y_train, epochs=EPOCH, batch_size=BATCH_SIZE,
                     validation_data=(X_test, Y_test), callbacks=[csv_logger])
print ('Optimization Finished!')
print ('')


history.history.keys()

# output
# save training/validation history
pd.DataFrame (history.history).to_csv ( 'rts/history_MFCC_seqlen_{}'.format (SEQ_LEN) + '.csv')


""" loss plot """
def plot_loss(epochs, loss, val_loss, class_category = None):
    plt.plot(epochs, loss, '--', label= 'Training Loss',lw =2)
    plt.plot(epochs, val_loss, '-', label= 'Validation Loss',lw =2)
    plt.title('training and validation loss - {}'.format(class_category), fontsize=18)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout ()


""" accuracy plot """
def plot_accuracy(epochs, acc, val_acc, class_category = None):
    plt.plot (epochs, acc, '--', label='Training Accuracy', lw=2)
    plt.plot (epochs, val_acc, '-', label='Validation Accuracy', lw=2)
    plt.title ('training and validation accuracy - {}'.format(class_category), fontsize=18)
    plt.xlabel ('Epochs', fontsize=14)
    plt.ylabel ('Accuracy', fontsize=14)
    plt.legend (fontsize=12)
    plt.tight_layout ()
    

""" confusion matrix plot """
def plot_confusion_matrix(cm, classes,normalize=False,
                         title='Confusion matrix',
                      cmap=plt.cm.PuBu):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')    
        print(cm)    
    plt.title(title, fontsize = 16)    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.clim(0, 1)
    plt.ylabel('True label', fontsize =12)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.tight_layout ()
    
##################################################
# save loss
loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['acc']
val_acc = history.history['val_acc']

# save loss plot
plt.figure ()
plot_loss (history.epoch, loss, val_loss)
plt.savefig (os.path.join ('rts/' + 'loss_MFCC_{}'.format (SEQ_LEN) + '.pdf'),
             bbox_inches="tight")


# evaluate on hold-out test data
score = model.evaluate(X_test, [Y_test], verbose=0)
print(score)
print('[Test] loss:', score[0])
print('[Test] accuracy:', score[1])

# get prediction on hold-out test data
Y_test_predict = model.predict(X_test)
Y_test_predict = np.argmax(Y_test_predict,axis=1) # convert one-hot encoded vector into 1d vector 
print(Y_test_predict.shape)

# calculate/plot confusion matrix -
confusion_matrix = metrics.confusion_matrix(np.argmax(Y_test,axis=1), Y_test_predict)
plt.figure()
plot_confusion_matrix(confusion_matrix, classes=emotion_list,normalize=True)
plt.savefig(os.path.join( 'rts/confusionmat_MFCC_{}'.format(SEQ_LEN)+ '.pdf'), bbox_inches="tight")

with open ('rts/' + 'summary_MFCC.csv', 'w') as txtfile:
    print ('summary : seqence length {}'.format(SEQ_LEN), file=txtfile)
    print('[Test] loss:', score[0], file= txtfile)
    print('[Test] accuracy:', score[1], file = txtfile)
    