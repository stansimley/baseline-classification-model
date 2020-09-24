from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import  GlobalAveragePooling2D, Dense, LeakyReLU
from keras import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import load_model
from keras import regularizers

''' number of classes, image size '''
NB_CLASS=10
IM_WIDTH=224
IM_HEIGHT=224

''' ***insert path to folder '''
train_root = 'path/to/train'
batch_size=32 #no. of I/P before updating weights
EPOCH=10 #complete rounds


''' data augmentation and validation split'''
#keras.preprocessing.image.ImageDataGenerator: Generate batches of tensor image data with real-time data augmentation (random modifications, e.g. width shift of random value).
#The data will be looped over (in batches).
train_datagen = ImageDataGenerator(
  width_shift_range=0.05, #shift left and right
  height_shift_range=0.05, #shift up and down
  rotation_range=45, #rotate ___ degrees
  brightness_range = None ,#range: [min,max], default = 1
  shear_range=0, #shear: shift one part of image in one direction and opposite side in the other
  zoom_range=0,#[min, max], default = 1
  channel_shift_range = 0,  #seems to shift all channels => brightness? lol
  horizontal_flip=True,
  vertical_flip=True,
  rescale = 1./255, #normalize RGB values in array to be [0,1]
  validation_split = 0.3 #split data
  )

# .flow_from_directory:Takes the path to a directory & generates batches of augmented data. (goes into sub directories)
train_generator = train_datagen.flow_from_directory(
  train_root, #directory
  target_size=(IM_WIDTH, IM_HEIGHT), #resize
  batch_size=batch_size,
  shuffle=True, #shuffle data => prevent overfitting
  subset='training'
)

''' create seperate datagen and generator if val data not in same folder'''
validation_generator = train_datagen.flow_from_directory(
    train_root, # same directory as training data
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    subset='validation') # set as validation data


#define base model, specifying arguments: weights, I/P tensor, pooling, no. of classes
#I/P for model: augmented data + pretrained layers
input_tensor = Input(shape=(IM_WIDTH, IM_HEIGHT, 3))

base_model = MobileNetV2(include_top=False, alpha = 0.35, weights = 'imagenet', input_tensor=input_tensor, pooling=None, classes=NB_CLASS) #include_top: T/F for including original classifier, weights: from somewhere (imagenet) or none, input_tensor = I/P

x = base_model.output
x = GlobalAveragePooling2D()(x) #average pooling prevents the network from learning the image structures such as edges and textures. And also translational invariance
x = Dense(1000, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-5))(x)
x = LeakyReLU(alpha=0.3)(x)
predictions = Dense(NB_CLASS, activation='softmax')(x) #add logistic layer for the classes:

model = Model(inputs=base_model.input, outputs=predictions)

#transfer learning
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=1, steps_per_epoch=train_generator.n/batch_size, validation_steps=validation_generator.n/batch_size) #class_weight=class_weight) #.n for number of samples in dataset, one epoch = one round of training, steps = no. of updates/epoch (or no. of batches trained) == total no. of samples/batch size

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy']) #Stochastic Gradient Descent
checkpoint = ModelCheckpoint("model_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1) #save model w weights only, choosing the one w the highest val acc
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto') #patience = 10 => stop if val acc does not improve after 10 epochs
history = model.fit_generator(generator=train_generator, validation_data = validation_generator, epochs=EPOCH,
steps_per_epoch=train_generator.n/batch_size, validation_steps=validation_generator.n/batch_size, callbacks=[checkpoint, early])
