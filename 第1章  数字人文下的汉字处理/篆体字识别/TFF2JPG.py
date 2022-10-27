import os
import time

from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont

'''
字体转图片
'''


def char_to_img(all_chara, img_dir, uniMap, font, img_size):
    i = 0
    for chara in all_chara:
        # 判断是否存在该字
        if ord(chara) in uniMap:
            # 新建长宽为300像素，背景色为白色的画布对象
            im = Image.new("RGB", (img_size, img_size), "white")
            draw = ImageDraw.Draw(im)
            # 从画布的坐标（0, 0）处绘制黑色汉字文本
            draw.text((0, 0), chara, fill="#000", font=font)
            # 获取图像中非零区域边界并裁剪
            im = im.crop(im.getbbox())
            # 保存汉字图像
            if not os.path.exists(img_dir + "/" + chara + "/"):
                os.mkdir(img_dir + "/" + chara + "/")
            save_path = img_dir + "/" + chara + "/" + str(len(os.listdir(img_dir + "/" + chara + "/"))) + ".png"
            im.save(save_path)


if __name__ == '__main__':
    start = time.clock()
    TTF_DIR = "ALL-Font"  # 存放.tff字体文件夹
    img_dir = "data"  # 生成图片存储路径
    img_size = 30  # 生成图片存储路径

    # 判断是否存在文件夹，若否，则创建
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # 选取需要保存的汉字
    all_chara = [chr(i) for i in range(19968,26000)]

    # 遍历每一个.tff字体文件
    for each_font in os.listdir(TTF_DIR):
        ttf_path = TTF_DIR + "/" + each_font
        print("********" + ttf_path + "*****************")
        # 创建int型unicode编码与字符映射表
        fontmap = TTFont(ttf_path)
        uniMap = fontmap['cmap'].tables[0].ttFont.getBestCmap()
        # 加载并创建指定大小的字体对象
        font = ImageFont.truetype(ttf_path, img_size)
        char_to_img(all_chara, img_dir, uniMap, font, img_size)
    print('time spent: {}'.format(time.clock()-start))


