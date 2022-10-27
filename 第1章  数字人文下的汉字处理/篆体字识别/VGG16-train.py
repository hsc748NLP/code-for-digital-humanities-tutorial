import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam


def add_input_top_model(base_model, class_num, input_shape):
    preprocessinput = preprocess_input

    inputs = tf.keras.Input(shape=input_shape)
    x = preprocessinput(inputs)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    # 若为2分类则
    if class_num == 2:
        outputs = Dense(1)(x)  # logit
    else:
        outputs = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


'''这里应该按照项目的需要进行设置'''


def model_compile(model, learning_rate=0.001):
    class_num = model.output.shape[1]
    if class_num == 2:
        # 如果是二分类模型
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss=BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model



'''一、 构建模型'''
'''注意下面几个变量根据需要修改！！！'''
IMG_SIZE = (224, 224)  # 此处为自己设置的图片的SIZE
IMG_SHAPE = IMG_SIZE + (3,)  # 根据设置的图片形状得到 —— （160, 160, 3）
NUM_CLASSES = len(os.listdir("data/"))  # 此处为需要进行分类的类别数量
BATCH_SIZE = 32  # 一个BATCH的大小

train_path = "data/"  # 训练集数据路径
# test_dir = "" #测试集数据路径
# valiation_dir = "" #验证集数据路径

# 训练得到的模型路径
save_path = "Ret_Model/"

'''1.1 构建预训练模型'''
print("BASE MODE:")
base_model = VGG16(input_shape=IMG_SHAPE, include_top=False)
# Let's take a look at the base model architecture
print(base_model.summary())

'''1.2 微调所需！！！'''
# base_model.trainable = True
# fine_tune_at = 16 #表明从第十六层开始重新进行训练
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False
# print(len(base_model.trainable_variables))

'''1.2 添加顶层分类器 & 输入层'''
print("ADD CLS TOP LAYER & INPUT LAYER:")
final_model = add_input_top_model(base_model, NUM_CLASSES, IMG_SHAPE)
print(final_model.summary())
# print(len(final_model.trainable_variables))

'''1.3 Compile'''
model = model_compile(final_model)
# model.summary()


'''二. 读入数据'''
# 注意修改路径

# 图像增强
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input,
                                                            # rotation_range=30,
                                                            # width_shift_range=0.2,
                                                            # height_shift_range=0.2,
                                                            # shear_range=0.2,
                                                            # zoom_range=0.2,
                                                            # horizontal_flip=True,
                                                            validation_split=0.2
                                                            )

train_generator = train_gen.flow_from_directory(train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                subset="training")
val_generator = train_gen.flow_from_directory(train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                              subset="validation")

# 注意修改路径
# 非图像增强
# train_dataset = image_dataset_from_directory(train_path,
#                       shuffle=True,
#                       batch_size=BATCH_SIZE,
#                       image_size=IMG_SIZE)
# 注意修改路径
# test_dataset = image_dataset_from_directory(test_dir,
#                       shuffle=True,
#                       batch_size=BATCH_SIZE,
#                       image_size=IMG_SIZE)
# 注意修改路径
# validation_dataset = image_dataset_from_directory(validation_dir,
#                     shuffle=True,
#                     batch_size=BATCH_SIZE,
#                     image_size=IMG_SIZE)

'''三. 训练模型'''
# 可以添加EarlyStopping——param: callback=[]
# 若划分了validationset,在fit时记得添加_ param: validation_data=
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=3)
history = model.fit(train_generator, validation_data=val_generator, epochs=30, callbacks=[callback])

model.save(save_path)



