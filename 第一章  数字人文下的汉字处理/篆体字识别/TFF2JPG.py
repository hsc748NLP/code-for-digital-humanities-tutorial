import os

from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont
from tqdm import tqdm


def char_to_img(all_chara, img_dir, uniMap, font, img_size):
    """借由汉字列表（all_chara）中的汉字，生成由指定字体构成的图片"""
    for chara in tqdm(all_chara, desc='正在生成字体图片'):
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
                os.makedirs(img_dir + "/" + chara + "/")
            save_path = img_dir + "/" + chara + "/" + str(len(os.listdir(img_dir + "/" + chara + "/"))) + ".png"
            im.save(save_path)


if __name__ == '__main__':
    TTF_DIR = "./ALL-Font"
    img_dir = "./data"
    img_size = 300  # 生成图片像素大小

    # 判断是否存在文件夹，若不存在，则创建
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    all_chara = [chr(i) for i in range(19968, 26000)]  # 5000个汉字

    for each_font in os.listdir(TTF_DIR):
        # ttf 字体文件存储路径
        ttf_path = TTF_DIR + "/" + each_font
        print("Processing:", ttf_path)
        # 创建Int型unicode拜尼马与字符映射表
        fontmap = TTFont(ttf_path)
        uniMap = fontmap['cmap'].tables[0].ttFont.getBestCmap()
        # 加载并创建指定大小的字体对象
        font = ImageFont.truetype(ttf_path, img_size)
        char_to_img(all_chara, img_dir, uniMap, font, img_size)
