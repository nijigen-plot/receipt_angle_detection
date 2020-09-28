# receipt_angle_detection
レシートを模擬した画像を生成し、角度を学習させてレシートの角度を検出する

## ファイル説明

### make_image.ipynb

レシートを模した角度45~-45度の長方形と背景にノイズを追加した白黒50x50の画像を35万枚生成します。

```python
def makeRectangle(l, w, theta, offset=(0,0)):
    c, s = math.cos(theta), math.sin(theta)
    rectCoords = [(l/2.0, w/2.0), (l/2.0, -w/2.0), (-l/2.0, -w/2.0), (-l/2.0, w/2.0)]
    return [(c*x-s*y+offset[0], s*x+c*y+offset[1]) for (x,y) in rectCoords]

L=50; W=50

random_color_base = np.random.randint(0, 255) # MAX 254
random_color_rectangle = np.random.randint(random_color_base + 1, 256) #MAX 255 少なくともbaseカラーよりは白い

image = Image.new("L", (L, W), random_color_base ) #8bit 白黒 0 = 黒 255 = 白
image_1 = np.array(image, dtype=np.uint8)
pts_x = np.random.randint(0, 50, 800)
pts_y = np.random.randint(0, 50, 800)

for i in range(len(pts_y)):
    image_1[(pts_y[i], pts_x[i])] = np.random.randint(0, 256)
image_2 = Image.fromarray(image_1)
draw = ImageDraw.Draw(image_2)

angle = np.random.randint(-45, 46)
size = np.random.randint(70, 131) / 100
vertices = makeRectangle(12 * size, 40 * size, angle *math.pi/180, offset=(L/2, W/2))
draw.polygon(vertices, fill= random_color_rectangle)
image_2.save(f"test.png")
Image_('test.png')
```

![画像例](https://user-images.githubusercontent.com/65853436/94451461-62b57900-01e9-11eb-851e-1bed881b86c4.png)

### keras_angle_learning.ipynb

make_image.ipynbにて作成した画像を、kerasを用いて畳み込みニューラルネットワークを構築、学習します。

```python
#層内容
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(128, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, input_shape=input_shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(learning_rate=0.0001),
             metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                           patience=3,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr=0.0001)
datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
model.summary()

datagen.fit(X_train)

h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                       epochs = epochs, validation_data = (X_valid, Y_valid),
                       verbose = 1, steps_per_epoch = X_train.shape[0] // batch_size,
                       callbacks=[learning_rate_reduction])
```


### predict_receipt_angle.ipynb

実際にスマートフォンで撮影したレシート画像の角度を分類します。
生成した画像で、撮影したレシート画像の角度を分類することは、現状うまくできていません。

![ダウンロード (5)](https://user-images.githubusercontent.com/65853436/94454680-21bf6380-01ed-11eb-9946-12c34aa82dc1.png)

### receipt_photo

レシート画像です。
