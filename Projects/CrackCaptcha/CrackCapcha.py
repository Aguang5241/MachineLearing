import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as tf
from skimage.measure import label, regionprops


def main():
    # 生成验证码
    def create_captcha(text, shear=0, size=(100, 24), scale=1):
        # 使用字母L来生成一张黑白图像，为`ImageDraw`类初始化一个实例
        im = Image.new('L', size, 'black')
        draw = ImageDraw.Draw(im)
        # 指定验证码文字所使用的字体
        font = ImageFont.truetype(
            font=r'C:\Users\75922\AppData\Local\Microsoft\Windows\Fonts\Coval-Black.ttf', size=22)
        draw.text((2, 2), text, fill=1, font=font)
        # PIL图像转换为`numpy`数组，以便用`scikit-image`库为图像添加错切变化效果
        image = np.array(im)
        affine_tf = tf.AffineTransform(shear=shear)
        image = tf.warp(image, affine_tf)
        return image / image.max()

    # 分割图像，返回小图像列表
    def segment_image(image):
        # scikit-image中的label函数能找出图像中像素值相同且又连接在一起的像素块
        label_image = label(image > 0)
        subimages = []
        for region in regionprops(label_image):
            start_x, start_y, end_x, end_y = region.bbox
            subimages.append(image[start_x:end_x, start_y:end_y])
        if len(subimages) == 0:
            return [image,]
        return subimages

    image = create_captcha('GENE', shear=0.2)
    plt.imshow(image, cmap='Greys')
    subimages = segment_image(image)
    f, axes = plt.subplots(1, len(subimages), figsize=(10, 3))
    for i in range(len(subimages)):
        axes[i].imshow(subimages[i], cmap="gray")
    plt.show()


if __name__ == '__main__':
    main()
