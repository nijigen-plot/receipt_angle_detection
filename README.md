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

### predict_receipt_angle.ipynb

実際にスマートフォンで撮影したレシート画像の角度を分類します。

### receipt_photo

レシート画像です。
