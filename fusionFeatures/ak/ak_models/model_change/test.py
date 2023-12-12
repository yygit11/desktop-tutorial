from keras import Sequential
from keras.layers import Dense, Input, BatchNormalization, ReLU
from keras.models import load_model
import autokeras as ak
from keras.models import Model

# 模型搭建
model = load_model("./FD5.h5")
model.summary()
config = model.get_config()
# print(type(config))
for key in config.values():
    print('values = {}'.format(key))
# model1 = Sequential.from_config(config)
# model1.load_weights("./111.h5", by_name=True)

# 手动搭建
# inputs = Input(shape=(739,))
# x = BatchNormalization(axis=1)(inputs)
# x = Dense(32)(x)
# x = ReLU(x)
# x = Dense(32)(x)
# x = ReLU(x)
# predictions = Dense(1, activation='softmax')(x)
# model = Model(inputs=inputs, outputs=predictions)
# model.summary()
# model = Sequential()
# model.add(Input(shape =(739,) ))
# model.add(Dense(32, name='dense'))
# model.add(Dense(32, name='dense_1'))
# model.add(Dense(1, name='dense_2'))
# model.summary()

# 权重保存
# model.save_weights('./111.h5')
# a = model.load_weights('./111.h5')
# print(a)
# model.load_weights("./FD5.h5", by_name=True)

# 查看权重
# print(model.layers[2])
# print(model.layers[2].get_weights()[0].shape)
# print(model.layers[2].get_weights()[1].shape)
# print(model.layers[2].get_weights())
