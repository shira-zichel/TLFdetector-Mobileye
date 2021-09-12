#%reload_ext autoreload
#%autoreload 2
#%autosave 120
#%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
from os.path import join
#from google.colab import drive
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, MaxPooling2D, BatchNormalization, Activation, Conv2D
import seaborn as sbn
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
def load_tfl_data(data_dir, crop_shape=(81,81)):
    images = np.memmap(join(data_dir,'data.bin'), mode='r', dtype=np.uint8).reshape([-1]+list(crop_shape) +[3])
    labels = np.memmap(join(data_dir,'labels.bin'), mode='r', dtype=np.uint8)
    return {'images':images,'labels':labels}

def viz_my_data(images,labels, predictions=None, num=(5,5), labels2name= {0:'No TFL',1:'green TFL',2:'red TFL'}):
    assert images.shape[0] == labels.shape[0]
    assert predictions is None or predictions.shape[0] == images.shape[0]
    h = 5
    n = num[0]*num[1]
    ax = plt.subplots(num[0],num[1],figsize=(h*num[0],h*num[1]),gridspec_kw={'wspace':0.05},squeeze=False,sharex=True,sharey=True)[1]#.flatten()
    idxs = np.random.randint(0,images.shape[0],n)
    for i,idx in enumerate(idxs):
        ax.flatten()[i].imshow(images[idx])
        # title = labels2name[labels[idx]]
        title =""
        if predictions is not None :
          pre0 = predictions[idx][0]*100
          pre1 = predictions[idx][1]*100
          pre2 = predictions[idx][2]*100
          pre0 = round(pre0,4)
          pre1 = round(pre1,4)
          pre2 = round(pre2,4)
          title += ' Prediction:NO {0} ,G {1} ,R {2}'.format(pre0,pre1,pre2)
          # title += ' Prediction: {0} %'.format(pre)

        ax.flatten()[i].set_title(title)
    # root = './'  #this is the root for your val and train datasets
    data_dir = r"C:\לימודים\val"
    datasets = {
        'val': load_tfl_data(join(data_dir, 'val')),
        'train': load_tfl_data(join(data_dir, 'train')),
    }
    for k, v in datasets.items():
        print('{} :  {} 0/1 split {:.1f} %'.format(k, v['images'].shape, np.mean(v['labels'] == 1) * 100))

    viz_my_data(num=(6, 6), **datasets['val'])

    #drive.mount('/content/drive')

    #define the model used for training


def tfl_model():
        input_shape = (81, 81, 3)

        model = Sequential()

        def conv_bn_relu(filters, **conv_kw):
            model.add(Conv2D(filters, use_bias=False, padding='same', kernel_initializer='he_normal', **conv_kw))

            model.add(BatchNormalization())
            model.add(Activation('relu'))

        def dense_bn_relu(units):
            model.add(Dense(units, use_bias=False, kernel_initializer='he_normal'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))

        def spatial_layer(count, filters):
            for i in range(count):
                conv_bn_relu(filters, kernel_size=(3, 3))
            conv_bn_relu(filters, kernel_size=(3, 3), strides=(2, 2))

        conv_bn_relu(32, kernel_size=(3, 3), input_shape=input_shape)
        spatial_layer(1, 32)
        spatial_layer(2, 64)
        spatial_layer(2, 96)

        model.add(Flatten())
        dense_bn_relu(96)
        model.add(Dense(3, activation='softmax'))
        return model

m = tfl_model()
m.summary()
    #train

data_dir = r"C:\לימודים\val"
datasets = {
        'val': load_tfl_data(join(data_dir, 'val')),
        'train': load_tfl_data(join(data_dir, 'train')),
    }
    # prepare our model
m = tfl_model()
m.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])

train, val = datasets['train'], datasets['val']
# train it, the model uses the 'train' dataset for learning. We evaluate the "goodness" of the model, by predicting the label of the images in the val dataset.
history = m.fit(train['images'], train['labels'], validation_data=(val['images'], val['labels']), epochs=100)

# compare train vs val acccuracy,
# why is val_accuracy not as good as train accuracy? are we overfitting?
epochs = history.history
epochs['train_accuracy'] = epochs['accuracy']
plt.figure(figsize=(10, 10))
for k in ['train_accuracy', 'val_accuracy']:
    plt.plot(range(len(epochs[k])), epochs[k], label=k)

plt.legend();

predictions = m.predict(val['images'])
sbn.distplot(predictions[:, 0]);
predicted_label = np.argmax(predictions, axis=-1)

print('accuracy:', np.mean(predicted_label == val['labels']))
# code copied from the training evaluation:


viz_my_data(num=(4,5),predictions=predictions,**val);
m.save("model.h5")
loaded_model = load_model("model.h5")
l_predictions = loaded_model.predict(val['images'])
sbn.distplot(l_predictions[:, 0]);

l_predicted_label = np.argmax(l_predictions, axis=-1)
print('accuracy:', np.mean(l_predicted_label == val['labels']))

def load_test_data(data_dir, crop_shape=(81, 81)):
    images = np.memmap(join(data_dir, 'data.bin'), mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
    return {'images': images}

def viz_test_data(images, predictions=None, num=(5, 5), labels2name={0: 'No TFL', 1: 'green TFL', 2: 'red TFL'}):

        assert predictions is None or predictions.shape[0] == images.shape[0]
        h = 5
        n = num[0] * num[1]
        ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05}, squeeze=False,sharex=True, sharey=True)[1]  # .flatten()
        idxs = np.random.randint(0, images.shape[0], n)
        for i, idx in enumerate(idxs):
            ax.flatten()[i].imshow(images[idx])
            title = ""
            if predictions is not None:
                pre_non = predictions[idx][0] * 100
                pre_green = predictions[idx][1] * 100
                pre_red = predictions[idx][2] * 100
                pre_non = round(pre_non, 4)
                pre_green = round(pre_green, 4)
                pre_red = round(pre_red, 4)
                title += ' Prediction:NO {0} ,G {1} ,R {2}'.format(pre_non, pre_green, pre_red)
                # title += ' Prediction: {0} %'.format(pre)

            ax.flatten()[i].set_title(title)

    # root = './'  #this is the root for your val and train datasets
data_dir = r"C:\לימודים\val"
datasets = {
        'test': load_test_data(join(data_dir, 'test')),

    }
for k, v in datasets.items():
        print('{} :  {} 0/1'.format(k, v['images'].shape))

viz_test_data(num=(6, 6), **datasets['test'])
test = datasets['test']
test_predictions = m.predict(test['images'])

viz_test_data(num=(4, 4), predictions=test_predictions, **test);