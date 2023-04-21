from removebg import RemoveBg
import os

rmbg = RemoveBg("UYnojh5H1p6FRyg8hrooJjTv", "erroe.log")
path = '%s/picture' % os.getcwd()
pathout = '%s/out' % os.getcwd()
for pic in os.listdir(path):
    rmbg.remove_background_from_img_file("%s\%s" % (path, pic))

# 需要pip install removebg
# 这个每个月只能免费抠50张，UYnojh5H1p6FRyg8hrooJjTv 是覃自己注册的一个密匙
# path是存放输入图片的路径，在py文件同级目录下，输出也会在这个目录下，输出的名称为“原图片名称+_no_bg.png”
