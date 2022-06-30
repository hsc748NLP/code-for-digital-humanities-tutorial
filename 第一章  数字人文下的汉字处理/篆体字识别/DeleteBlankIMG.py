import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def clean_blank(image_path):
    # 解决全白图片的方案：
    for each_cls in os.listdir(img_dir):
        dir_path = os.path.join(img_dir, each_cls)
        for each_img in os.listdir(dir_path):
            image_path = os.path.join(dir_path, each_img)
            # 1、加载图片
            image = tf.keras.preprocessing.image.load_img(image_path)
            # 将图片转化为numpy的ndarray类型
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            # 2、判断上面的 input_arr的每个元素是否相同 —— 每个像素的RGB数值是否相同
            ele = input_arr[0]
            for i in range(len(input_arr)):
                if ele.all() != input_arr[i].all():
                    # 非空白图片，跳出循环
                    break
                if i == len(input_arr) - 1:
                    # 空白图片，删除
                    os.remove(image_path)


if __name__ == '__main__':
    img_dir = "./data"  # 生成的字体图片路径

    for each_imge in tqdm(os.listdir(img_dir)):
        clean_blank(os.path.join(img_dir, each_imge))
