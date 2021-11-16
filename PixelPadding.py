import PIL.Image
import cv2
import numpy as np
import os
from PIL import Image
#import matplotlib.pyplot as plt
import  time
import sys
import  subprocess

#C++ 调用本脚本
#  FWindowsPlatformProcess::CreateProc(TEXT("D:\\ZC\\dist\\1.exe"), TEXT(""), true, false, false, nullptr, -1, nullptr, nullptr);

#  CMD 的命令


# py文件名 参数1 参数2 参数3
# 参数1：输入图片  参数2：输出图片以及合成全景图的位置 参数3：krpano软件的位置



# 1.py D:\ZC D:\OutPut_Test D:/Krpano/krpano-1.19-pr13/1.bat
# 1.exe D:\ZC D:\OutPut_Test D:/Krpano/krpano-1.19-pr13/1.bat

#   1.exe D:\ZC\PNG D:\OutPut_Test D:/Krpano/krpano-1.19-pr13/1.bat new_name
#   PixelPadding.py D:\ZC\PNG D:\OutPut_Test D:/Krpano/krpano-1.19-pr13/1.bat new_name
#   PixelPadding.exe D:\ZC\PNG D:\OutPut_Test D:/Krpano/krpano-1.19-pr13/1.bat new_name

input_img_path = sys.argv[1]
#          D:\ZC
output_img_path = sys.argv[2]

#       D:\OutPut_Test

krpano_path= sys.argv[3]
#        D:/Krpano/krpano-1.19-pr13/1.bat


sphere_name= sys.argv[4]


#path="D:/ZC"
path=input_img_path

def PNG_JPG(PngPath):
    img = cv2.imread(PngPath, 0)
    #img = cv.imread(PngPath)
    w, h = img.shape[::-1]
    infile = PngPath
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    #img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    #img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=70)
            os.remove(PngPath)
        else:
            img.convert('RGB').save(outfile, quality=70)
            os.remove(PngPath)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)

def Zc_Png2JPG(name):
    img = Image.open(input_img_path+"\\"+name)  # 打开图片
    img = img.convert('RGB')
    img.save(output_img_path+"\\"+name.replace('.png','.jpg'))


    # 注意这里是因为还没有处理下面4条缝隙
    if name=="down.jpg":
        pass
    else:
        os.remove(input_img_path+"\\"+name)


# PNG_JPG(path+"/front.png")
# PNG_JPG(path+"/back.png")
# PNG_JPG(path+"/left.png")
# PNG_JPG(path+"/right.png")
# PNG_JPG(path+"/top.png")
# PNG_JPG(path+"/down.png")

Zc_Png2JPG("front.png")
Zc_Png2JPG("back.png")
Zc_Png2JPG("right.png")
Zc_Png2JPG("left.png")
Zc_Png2JPG("top.png")
Zc_Png2JPG("down.png")

print(" PNG转换成JPG结束  ")



def Zc_PixelPadding_Horizontal(first_img_name,second_img_name,input_path,output_path,fx,green_line,fuzzy):

    '''


    :param first_img_name:  the left pic
    :param second_img_name: the right pic
    :param input_path:  lazy to explain
    :param output_path: lazy to explain,  concentrate on!!! you’d better wirie input_path as output_path,
            beacuse we usually   input pic what we out  output before
    :param fx:   same to "feng xi"  means use fx range to merge and fix up pixel
    :param green_line:   draw greenline
    :param fuzzy:     a type of  convolution  , it doesn't work well
    :param bilateralFilter:   a type of  convolution, it doesn't work well
    :return:
    '''

    '''
    说人话：
    :param first_img_name:  合成两张图，左边的那张
    :param second_img_name: 合成两张图，右边的那张
    :param input_path:     
    :param output_path: 
    :param fx:   缝隙，注意这个是用来合成像素的缝隙，不是原有缝隙，就是用这么大一个范围去取色，哦对，取色范围
    :param green_line:    要不要画辅助线
    :param fuzzy:     要不要用  模糊滤波   建议不用，比较坑
    :param bilateralFilter:   要不要用  双边滤波   建议不用，虽然不吭但是效果不明显，对了这两个滤波可以用时开启，但建议别用
    :return: 
    
    '''
    bilateralFilter=0
    threshold0 = 50  #差值，单个颜色超过这个值就会跳过
    threshold1 = 0.04 #0.02
    threshold2 = 80  #颜色总和，如果小于它就会跳过

    fx=fx
    n = int (fx / 2)
    _range=int (fx / 2)
    enhance = 1

    add_light_range = 0     #z追加缝隙光照范围
    light = 1  #追加的亮度的一个因子
    c = 0       #相加项


    balance_range=0

    one=True


    second_img = Image.open(input_path+'/'+second_img_name)  # 打开图片

    pix = second_img.load()  # 导入像素
    width = second_img.size[0]  # 获取宽度
    height = second_img.size[1]  # 获取长度

    ##构建了两个矩阵，用来存储、读取图片的像素值
    first_img_zeros=np.zeros((width,height,3))
    second_img_zeros=np.zeros((width,height,3))
    #(1024,1024,3)




    #可以理解为，我们要处理的宽度，left图片是处理右边，back是处理左边，不清楚就看看left.jpg和back.jpg
    Mask_X=int(    width/5    )

    Extra_Mask_x=int(Mask_X*1.2)

    for x in range(width-Extra_Mask_x,width):
        for y in range(height):
            r, g, b= second_img.getpixel((x, y))
            second_img_zeros[x][y][0]=r
            second_img_zeros[x][y][1]=g
            second_img_zeros[x][y][2]=b
            #back.putpixel((x, y), (int(r*1), int(g*1), int(b*1)))
    second_img = second_img.convert('RGB')

    ########################################读取back图片的像素

    #first_img = Image.open('2048/left.jpg')  # 打开图片
    first_img = Image.open(input_path+"/"+first_img_name)  # 打开图片

    width = first_img.size[0]  # 获取宽度
    height = first_img.size[1]  # 获取长度

    for x in range(Extra_Mask_x):
        for y in range(height):
            r, g, b= first_img.getpixel((x, y))
            #left_pixel.append((x, y, (r, g, b)))
            first_img_zeros[x][y][0]=r
            first_img_zeros[x][y][1]=g
            first_img_zeros[x][y][2]=b
            #left.putpixel((x, y), (int(r * 1), int(g * 1), int(b * 1)))



    first_img = first_img.convert('RGB')
    ######################################读取left图片的像素



    for y in range(1,height-1):

        skip_pixel = 5


        # first_img.putpixel((skip_pixel, y), (0, 0, 255))
        # second_img.putpixel((width - 1 - skip_pixel, y), (255, 0,0 ))

        for x in range(_range):  # 均色填充
            l_r, l_g, l_b = first_img.getpixel((x, y))
            b_r, b_g, b_b = second_img.getpixel((width - x-1 , y))

            # l_r1, l_g1, l_b1 = left.getpixel((x, y-1))
            # b_r1, b_g1, b_b1 = back.getpixel((width - x-1 , y-1))

            l_r1, l_g1, l_b1 = first_img_zeros[x,y-1,0],first_img_zeros[x,y-1,1],first_img_zeros[x,y-1,2],
            b_r1, b_g1, b_b1 = second_img_zeros[width-x-1,y-1,0],second_img_zeros[width-x-1,y-1,1],second_img_zeros[width-x-1,y-1,2],

            try:
                rate_r00=abs(l_r-l_r1) / l_r
                rate_g00=abs(l_g-l_g1) / l_g
                rate_b00=abs(l_b-l_b1) / l_b

                rate_r11=abs(b_r-b_r1) / b_r
                rate_g11=abs(b_g-b_g1) / b_g
                rate_b11=abs(b_b-b_b1) / b_b
            except:
                rate_r00,rate_g00,rate_b00=2,2,2
                rate_r11,rate_g11,rate_b11=2,2,2

            limit_rate00=(rate_r00+rate_g00+rate_b00)/3
            limit_rate11=(rate_r11+rate_g11+rate_b11)/3


            l_r2, l_g2, l_b2 = first_img.getpixel((x+skip_pixel, y))
            b_r2, b_g2, b_b2 = second_img.getpixel((width - (x+skip_pixel) - 1, y))


            r=int ((l_r2+b_r2)/2    *enhance  )# +   int(light_add*(_range-x)/_range)
            g=int  ((l_g2+b_g2)/2     *enhance  )# +   int(light_add*(_range-x)/_range)
            b=int ((l_b2+b_b2)/2   *enhance     )# +  int(light_add*(_range-x)/_range)


            if  limit_rate00<threshold1:
            #if  1:
                #continue
                first_img.putpixel((x, y), (r, g, b))



            elif l_r+l_g+l_b<120:

                first_img.putpixel((x, y), (l_r2, l_g2, l_b2))

            if  limit_rate11<threshold1:
            #if  1:
                #continue
                second_img.putpixel((width - 1 - x , y), (r, g, b))


            elif b_r+b_g+b_b<120:

                second_img.putpixel((width - 1 - x, y), (b_r2, b_g2, b_b2))


        #b = int(fx * 3)
        rgb_loss=[]
        for x in range(fx,fx*2):
            l_r, l_g, l_b = first_img.getpixel((x, y))
            b_r, b_g, b_b = second_img.getpixel((width - 1 - x, y))

            loss= int(      ( (l_r+l_g+l_b)-(b_r+b_g+b_b) )/3   )
            rgb_loss.append(loss)

        balance=int (sum(rgb_loss)/len(rgb_loss))

        for x in range(balance_range):  # 亮度平衡
            l_r, l_g, l_b = first_img.getpixel((x, y))
            b_r, b_g, b_b = second_img.getpixel((width - 1 - x, y))
            if (l_r + l_g + l_b > 100):
                l_r = l_r - int  ( (balance/2 ) * (balance_range-x)/balance_range    )
                l_g = l_g - int  ( (balance/2 ) * (balance_range-x)/balance_range    )
                l_b = l_b - int  ( (balance/2 ) * (balance_range-x)/balance_range    )

            else:
                pass
            if (b_r + b_g + b_b > 100):
                b_r = b_r +  int  ( (balance/2 ) * (balance_range-x)/balance_range    )
                b_g = b_g +  int  ( (balance/2 ) * (balance_range-x)/balance_range    )
                b_b = b_b +  int  ( (balance/2 ) * (balance_range-x)/balance_range    )

            else:
                pass

            first_img.putpixel((x, y), (l_r, l_g, l_b))
            second_img.putpixel((width - 1 - x, y), (b_r, b_g, b_b))

        r0,g0,b0=first_img.getpixel((0, y))
        r1,g1,b1=first_img.getpixel((1, y))

        r2,g2,b2=second_img.getpixel((width-1, y))
        r3,g3,b3=second_img.getpixel((width-2, y))

        r = int((r0 + r1 + r2 + r3) / 4)
        g = int((g0 + g1 + g2 + g3) / 4)
        b = int((b0 + b1 + b2 + b3) / 4)

        # r = int((r0  + r2 ) / 2)
        # g = int((g0  + g2 ) / 2)
        # b = int((b0 + b2 ) / 2)


        for x in range(4):  # 均色填充
            pass
            # first_img.putpixel((x,y),(r,g,b))
            # second_img.putpixel((width - 1 - x, y), (r,g,b))






        for x in range(add_light_range):  # 就只是增加亮度
            t_r, t_g, t_b = first_img.getpixel((x, y))
            d_r, d_g, d_b = second_img.getpixel((width-1-x, y))

            if(t_r+t_g+t_b>100):
                tr = t_r + int((add_light_range - x) / add_light_range * light)+c
                tg = t_g +int((add_light_range - x) / add_light_range * light)+c
                tb = t_b + int((add_light_range - x) / add_light_range * light)+c
            else:
                tr = t_r
                tg = t_g
                tb = t_b

            if (d_r + d_g + d_b > 100):
                dr = d_r + int((add_light_range - x) / add_light_range * light)+c
                dg = d_g + int((add_light_range - x) / add_light_range * light)+c
                db = d_b + int((add_light_range - x) / add_light_range * light)+c
            else:
                dr = d_r
                dg = d_g
                db = d_b


            first_img.putpixel((x, y), (tr, tg, tb))
            second_img.putpixel((width-1-x,y), (dr, dg, db))


    if(green_line):
        for y in range(height):
            first_img.putpixel((fx,y),(0,255,0))
            second_img.putpixel((width-1-fx,y),(0,255,0))



    first_img = first_img.convert('RGB')
    second_img = second_img.convert('RGB')

    #first_img.save("2048_processed/left.jpg")
    first_img.save(output_path+"/"+first_img_name)
    second_img.save(output_path+"/"+second_img_name)






    #使用双边滤波进行处理
    if bilateralFilter:

        first_img=cv2.imread(output_path+"/"+first_img_name)
        second_img=cv2.imread(output_path+"/"+second_img_name)


        first_img = cv2.bilateralFilter(first_img,9,75,75)
        second_img = cv2.bilateralFilter(second_img,9,75,75)

        cv2.imwrite(output_path+"/"+first_img_name,first_img)
        cv2.imwrite(output_path+"/"+second_img_name,second_img)



def Zc_PixelPadding_Top2(first_img_name, second_img_name, input_path, output_path, fx, green_line, fuzzy):
    '''


    :param first_img_name:  the left pic
    :param second_img_name: the right pic
    :param input_path:  lazy to explain
    :param output_path: lazy to explain,  concentrate on!!! you’d better wirie input_path as output_path,
            beacuse we usually   input pic what we out  output before
    :param fx:   same to "feng xi"  means use fx range to merge and fix up pixel
    :param green_line:   draw greenline
    :param fuzzy:     a type of  convolution  , it doesn't work well
    :param bilateralFilter:   a type of  convolution, it doesn't work well
    :return:
    '''

    '''
    说人话：
    :param first_img_name:  合成两张图，左边的那张
    :param second_img_name: 合成两张图，右边的那张
    :param input_path:     
    :param output_path: 
    :param fx:   缝隙，注意这个是用来合成像素的缝隙，不是原有缝隙，就是用这么大一个范围去取色，哦对，取色范围
    :param green_line:    要不要画辅助线
    :param fuzzy:     要不要用  模糊滤波   建议不用，比较坑
    :param bilateralFilter:   要不要用  双边滤波   建议不用，虽然不吭但是效果不明显，对了这两个滤波可以用时开启，但建议别用
    :return: 

    '''
    bilateralFilter = 0
    threshold0 = 50  # 差值，单个颜色超过这个值就会跳过
    threshold1 = 0.001  # 0.02
    threshold2 = 80  # 颜色总和，如果小于它就会跳过

    fx = fx
    n = int(fx / 2)
    _range = int(fx / 2)
    enhance = 1

    # 妈的，下面这俩玩意儿会出现竖线,以前是10,100
    light_add = 0
    light_range = 0

    # add_light_range =120      #z追加缝隙光照范围
    # light = 7   #追加的亮度的一个因子
    # c = 0       #相加项

    # add_light_range =50     #z追加缝隙光照范围
    add_light_range = 0     #z追加缝隙光照范围
    light = 1  #追加的亮度的一个因子
    c = 0      #相加项

    balance_range = 0

    one = True

    second_img = Image.open(input_path + '/' + second_img_name)  # 打开图片

    pix = second_img.load()  # 导入像素
    width = second_img.size[0]  # 获取宽度
    height = second_img.size[1]  # 获取长度

    ##构建了两个矩阵，用来存储、读取图片的像素值
    first_img_zeros = np.zeros((width, height, 3))
    second_img_zeros = np.zeros((width, height, 3))

    # 可以理解为，我们要处理的宽度，left图片是处理右边，back是处理左边，不清楚就看看left.jpg和back.jpg
    Mask_X = int(width / 5)

    Extra_Mask_x = int(Mask_X * 1.2)

    for x in range(width):
        for y in range(height):
            r, g, b = second_img.getpixel((x, y))
            second_img_zeros[x][y][0] = r
            second_img_zeros[x][y][1] = g
            second_img_zeros[x][y][2] = b
            # back.putpixel((x, y), (int(r*1), int(g*1), int(b*1)))
    second_img = second_img.convert('RGB')

    ########################################读取back图片的像素

    # first_img = Image.open('2048/left.jpg')  # 打开图片
    first_img = Image.open(input_path + "/" + first_img_name)  # 打开图片
    pix = first_img.load()  # 导入像素
    width = first_img.size[0]  # 获取宽度
    height = first_img.size[1]  # 获取长度

    for x in range(width):
        for y in range(fx*2):
            r, g, b = first_img.getpixel((x, y))
            # left_pixel.append((x, y, (r, g, b)))
            first_img_zeros[x][y][0] = r
            first_img_zeros[x][y][1] = g
            first_img_zeros[x][y][2] = b
            # left.putpixel((x, y), (int(r * 1), int(g * 1), int(b * 1)))

    first_img = first_img.convert('RGB')
    ######################################读取left图片的像素



    if first_img_name=="left.jpg":

        for constant in range(1, height - 1):

            skip_pixel = 2

            # first_img.putpixel((skip_pixel, y), (0, 0, 255))
            # second_img.putpixel((width - 1 - skip_pixel, y), (255, 0,0 ))

            for variable in range(_range):  # 均色填充

                #记住second 是 top

                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((variable, constant))


                f_r1, f_g1, f_b1 = first_img_zeros[constant-1,variable, 0], first_img_zeros[constant-1,variable, 1], first_img_zeros[constant-1,variable, 2],
                s_r1, s_g1, s_b1 = second_img_zeros[variable, constant - 1, 0], second_img_zeros[variable, constant - 1, 1], \
                                   second_img_zeros[variable, constant - 1, 2],



                try:
                    rate_r00 = abs(f_r - f_r1) / f_r
                    rate_g00 = abs(f_g - f_g1) / f_g
                    rate_b00 = abs(f_b - f_b1) / f_b

                    rate_r11 = abs(s_r - s_r1) / s_r
                    rate_g11 = abs(s_g - s_g1) / s_g
                    rate_b11 = abs(s_b - s_b1) / s_b
                except:
                    rate_r00, rate_g00, rate_b00 = 2, 2, 2
                    rate_r11, rate_g11, rate_b11 = 2, 2, 2

                limit_rate00 = (rate_r00 + rate_g00 + rate_b00) / 3
                limit_rate11 = (rate_r11 + rate_g11 + rate_b11) / 3


                f_r2, f_g2, f_b2 = first_img.getpixel((constant, variable+skip_pixel))
                s_r2, s_g2, s_b2 = second_img.getpixel((variable+skip_pixel,constant))

                r = int((f_r2 + s_r2) / 2 * enhance)  # +   int(light_add*(_range-x)/_range)
                g = int((f_g2 + s_g2) / 2 * enhance)  # +   int(light_add*(_range-x)/_range)
                b = int((f_b2 + s_b2) / 2 * enhance)  # +  int(light_add*(_range-x)/_range)


                if limit_rate00 < threshold1:
                    # if  1:
                    first_img.putpixel((constant, variable), (r, g, b))

                    # left.putpixel((x, y), (0, 0, 255))
                    # left.putpixel((x, y), (255, 0, 0))

                elif f_r + f_g + f_b < 120:

                    first_img.putpixel((constant, variable), (f_r2, f_g2, f_b2))
                    #pass
                    #first_img.putpixel((constant, variable), (0, 0, 0))
                    #print("这时候xy", (constant, variable))
                else:
                    first_img.putpixel((constant, variable), (f_r2, f_g2, f_b2))
                    #print("这时候x          y",(constant,variable))
                    #pass
                    #first_img.putpixel((constant, variable), (0, 0, 0))

                if limit_rate11 < threshold1:
                    # if  1:
                    second_img.putpixel((variable, constant), (r, g, b))

                    # back.putpixel((width - 1 - x, y), ( (0, 0, 255)))
                    # back.putpixel((width - 1 - x, y), (255, 0, 0))

                elif s_r + s_g + s_b < 120:

                    second_img.putpixel((variable, constant), (s_r2, s_g2, s_b2))


            rgb_loss = []
            for variable in range(fx, fx * 2):
                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((variable, constant))

                loss = int(((f_r + f_g + f_b) - (s_r + s_g + s_b)) / 3)
                rgb_loss.append(loss)

            balance = int(sum(rgb_loss) / len(rgb_loss))

            for variable in range(balance_range):  # 亮度平衡
                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((variable, constant))
                if (f_r + f_g + f_b > 100):
                    f_r = f_r - int((balance / 2) * (balance_range - variable) / balance_range)
                    f_g = f_g - int((balance / 2) * (balance_range - variable) / balance_range)
                    f_b = f_b - int((balance / 2) * (balance_range - variable) / balance_range)

                else:
                    pass
                if (s_r + s_g + s_b > 100):
                    s_r = s_r + int((balance / 2) * (balance_range - variable) / balance_range)
                    s_g = s_g + int((balance / 2) * (balance_range - variable) / balance_range)
                    s_b = s_b + int((balance / 2) * (balance_range - variable) / balance_range)

                else:
                    pass

                first_img.putpixel((constant, variable), (f_r, f_g, f_b))
                second_img.putpixel((variable, constant), (s_r, s_g, s_b))

            r0, g0, b0 = first_img.getpixel((constant, 0))
            r1, g1, b1 = first_img.getpixel((constant, 1))

            r2, g2, b2 = second_img.getpixel((0, constant))
            r3, g3, b3 = second_img.getpixel((1, constant))

            r = int((r0 + r1 + r2 + r3) / 4)
            g = int((g0 + g1 + g2 + g3) / 4)
            b = int((b0 + b1 + b2 + b3) / 4)

            # r = int((r0  + r2 ) / 2)
            # g = int((g0  + g2 ) / 2)
            # b = int((b0 + b2 ) / 2)

            for variable in range(2):  # 均色填充
                # pass
                first_img.putpixel((constant, variable), (r, g, b))
                second_img.putpixel((variable, constant), (r, g, b))



            for variable in range(add_light_range):  # 就只是增加亮度
                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((variable, constant))

                if (f_r + f_g + f_b > 100):
                    fr = f_r + int((add_light_range - variable) / add_light_range * light) + c
                    fg = f_g + int((add_light_range - variable) / add_light_range * light) + c
                    fb = f_b + int((add_light_range - variable) / add_light_range * light) + c
                else:
                    fr = f_r
                    fg = f_g
                    fb = f_b

                if (s_r + s_g + s_b > 100):
                    sr = s_r + int((add_light_range - variable) / add_light_range * light) + c
                    sg = s_g + int((add_light_range - variable) / add_light_range * light) + c
                    sb = s_b + int((add_light_range - variable) / add_light_range * light) + c
                else:
                    sr = s_r
                    sg = s_g
                    sb = s_b

                first_img.putpixel((constant, variable), (fr, fg, fb))
                second_img.putpixel((variable, constant), (sr, sg, sb))


    if first_img_name=="right.jpg":

        for constant in range(1, height - 1):

            skip_pixel = 2

            # first_img.putpixel((skip_pixel, y), (0, 0, 255))
            # second_img.putpixel((width - 1 - skip_pixel, y), (255, 0,0 ))

            for variable in range(_range):  # 均色填充

                #记住second 是 top

                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((width-1-variable, width-1-constant))


                f_r1, f_g1, f_b1 = first_img_zeros[constant-1,variable, 0], first_img_zeros[constant-1,variable, 1], first_img_zeros[constant-1,variable, 2],
                s_r1, s_g1, s_b1 = second_img_zeros[width-1-variable, width-1-constant + 1, 0], second_img_zeros[width-1-variable, width-1-constant + 1, 1], \
                                   second_img_zeros[width-1-variable, width-1-constant + 1, 2],



                try:
                    rate_r00 = abs(f_r - f_r1) / f_r
                    rate_g00 = abs(f_g - f_g1) / f_g
                    rate_b00 = abs(f_b - f_b1) / f_b

                    rate_r11 = abs(s_r - s_r1) / s_r
                    rate_g11 = abs(s_g - s_g1) / s_g
                    rate_b11 = abs(s_b - s_b1) / s_b
                except:
                    rate_r00, rate_g00, rate_b00 = 2, 2, 2
                    rate_r11, rate_g11, rate_b11 = 2, 2, 2

                limit_rate00 = (rate_r00 + rate_g00 + rate_b00) / 3
                limit_rate11 = (rate_r11 + rate_g11 + rate_b11) / 3


                f_r2, f_g2, f_b2 = first_img.getpixel((constant, variable+skip_pixel))
                s_r2, s_g2, s_b2 = second_img.getpixel((width-1-variable-skip_pixel,width-1-constant))

                r = int((f_r2 + s_r2) / 2 * enhance)  # +   int(light_add*(_range-x)/_range)
                g = int((f_g2 + s_g2) / 2 * enhance)  # +   int(light_add*(_range-x)/_range)
                b = int((f_b2 + s_b2) / 2 * enhance)  # +  int(light_add*(_range-x)/_range)

                if limit_rate00 < threshold1:
                    # if  1:
                    first_img.putpixel((constant, variable), (r, g, b))

                    # left.putpixel((x, y), (0, 0, 255))
                    # left.putpixel((x, y), (255, 0, 0))

                elif f_r + f_g + f_b < 120:

                    first_img.putpixel((constant, variable), (f_r2, f_g2, f_b2))
                    #pass
                    #first_img.putpixel((constant, variable), (0, 0, 0))
                    #print("这时候xy", (constant, variable))
                else:
                    #print("这时候x          y",(constant,variable))
                    first_img.putpixel((constant, variable), (f_r2, f_g2, f_b2))
                    #pass
                    #first_img.putpixel((constant, variable), (0, 0, 0))

                if limit_rate11 < threshold1:
                    # if  1:
                    second_img.putpixel((width-1-variable, width-1-constant), (r, g, b))

                    # back.putpixel((width - 1 - x, y), ( (0, 0, 255)))
                    # back.putpixel((width - 1 - x, y), (255, 0, 0))

                elif s_r + s_g + s_b < 120:

                    second_img.putpixel((width-1-variable, width-1-constant), (s_r2, s_g2, s_b2))


            rgb_loss = []
            for variable in range(fx, fx * 2):
                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((width-1-variable, width-1-constant))

                loss = int(((f_r + f_g + f_b) - (s_r + s_g + s_b)) / 3)
                rgb_loss.append(loss)

            balance = int(sum(rgb_loss) / len(rgb_loss))

            for variable in range(balance_range):  # 亮度平衡
                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((width-1-variable, width-1-constant))
                if (f_r + f_g + f_b > 100):
                    f_r = f_r - int((balance / 2) * (balance_range - variable) / balance_range)
                    f_g = f_g - int((balance / 2) * (balance_range - variable) / balance_range)
                    f_b = f_b - int((balance / 2) * (balance_range - variable) / balance_range)

                else:
                    pass
                if (s_r + s_g + s_b > 100):
                    s_r = s_r + int((balance / 2) * (balance_range - variable) / balance_range)
                    s_g = s_g + int((balance / 2) * (balance_range - variable) / balance_range)
                    s_b = s_b + int((balance / 2) * (balance_range - variable) / balance_range)

                else:
                    pass

                first_img.putpixel((constant, variable), (f_r, f_g, f_b))
                second_img.putpixel((width-1-variable, width-1-constant), (s_r, s_g, s_b))

            r0, g0, b0 = first_img.getpixel((constant, 0))
            r1, g1, b1 = first_img.getpixel((constant, 1))

            r2, g2, b2 = second_img.getpixel((width-1-0, width-1-constant))
            r3, g3, b3 = second_img.getpixel((width-1-1, width-1-constant))

            r = int((r0 + r1 + r2 + r3) / 4)
            g = int((g0 + g1 + g2 + g3) / 4)
            b = int((b0 + b1 + b2 + b3) / 4)

            # r = int((r0  + r2 ) / 2)
            # g = int((g0  + g2 ) / 2)
            # b = int((b0 + b2 ) / 2)

            for variable in range(2):  # 均色填充
                # pass
                first_img.putpixel((constant, variable), (r, g, b))
                second_img.putpixel((width-1-variable, width-1-constant), (r, g, b))






            for variable in range(add_light_range):  # 就只是增加亮度
                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((width-1-variable, width-1-constant))

                if (f_r + f_g + f_b > 100):
                    fr = f_r + int((add_light_range - variable) / add_light_range * light) + c
                    fg = f_g + int((add_light_range - variable) / add_light_range * light) + c
                    fb = f_b + int((add_light_range - variable) / add_light_range * light) + c
                else:
                    fr = f_r
                    fg = f_g
                    fb = f_b

                if (s_r + s_g + s_b > 100):
                    sr = s_r + int((add_light_range - variable) / add_light_range * light) + c
                    sg = s_g + int((add_light_range - variable) / add_light_range * light) + c
                    sb = s_b + int((add_light_range - variable) / add_light_range * light) + c
                else:
                    sr = s_r
                    sg = s_g
                    sb = s_b

                first_img.putpixel((constant, variable), (fr, fg, fb))
                second_img.putpixel((width-1-variable, width-1-constant), (sr, sg, sb))


    # if first_img_name=="back.jpg":
    #
    #     for constant in range(1, height - 1):
    #
    #         skip_pixel = 2
    #
    #         # first_img.putpixel((skip_pixel, y), (0, 0, 255))
    #         # second_img.putpixel((width - 1 - skip_pixel, y), (255, 0,0 ))
    #
    #         for variable in range(_range):  # 均色填充
    #
    #             #记住second 是 top
    #
    #             f_r, f_g, f_b = first_img.getpixel((constant, variable))
    #             s_r, s_g, s_b = second_img.getpixel((constant, variable))
    #
    #
    #             f_r1, f_g1, f_b1 = first_img_zeros[constant-1,variable, 0], first_img_zeros[constant-1,variable, 1], first_img_zeros[constant-1,variable, 2],
    #             s_r1, s_g1, s_b1 = second_img_zeros[constant-1,variable, 0], second_img_zeros[constant-1,variable, 1], \
    #                                second_img_zeros[constant-1,variable, 2],
    #
    #
    #
    #             try:
    #                 rate_r00 = abs(f_r - f_r1) / f_r
    #                 rate_g00 = abs(f_g - f_g1) / f_g
    #                 rate_b00 = abs(f_b - f_b1) / f_b
    #
    #                 rate_r11 = abs(s_r - s_r1) / s_r
    #                 rate_g11 = abs(s_g - s_g1) / s_g
    #                 rate_b11 = abs(s_b - s_b1) / s_b
    #             except:
    #                 rate_r00, rate_g00, rate_b00 = 2, 2, 2
    #                 rate_r11, rate_g11, rate_b11 = 2, 2, 2
    #
    #             limit_rate00 = (rate_r00 + rate_g00 + rate_b00) / 3
    #             limit_rate11 = (rate_r11 + rate_g11 + rate_b11) / 3
    #
    #
    #             f_r2, f_g2, f_b2 = first_img.getpixel((constant, variable+skip_pixel))
    #             s_r2, s_g2, s_b2 = second_img.getpixel((constant, variable+skip_pixel))
    #
    #             r = int((f_r2 + s_r2) / 2 * enhance)  # +   int(light_add*(_range-x)/_range)
    #             g = int((f_g2 + s_g2) / 2 * enhance)  # +   int(light_add*(_range-x)/_range)
    #             b = int((f_b2 + s_b2) / 2 * enhance)  # +  int(light_add*(_range-x)/_range)
    #
    #             if limit_rate00 < threshold1:
    #                 # if  1:
    #                 first_img.putpixel((constant, variable), (r, g, b))
    #
    #                 # left.putpixel((x, y), (0, 0, 255))
    #                 # left.putpixel((x, y), (255, 0, 0))
    #
    #             elif f_r + f_g + f_b < 120:
    #
    #                 #first_img.putpixel((constant, variable), (f_r2, f_g2, f_b2))
    #                 first_img.putpixel((constant, variable), (0, 0, 0))
    #                 #print("这时候xy", (constant, variable))
    #
    #
    #             if limit_rate11 < threshold1:
    #                 # if  1:
    #                 second_img.putpixel((constant, variable), (r, g, b))
    #
    #                 # back.putpixel((width - 1 - x, y), ( (0, 0, 255)))
    #                 # back.putpixel((width - 1 - x, y), (255, 0, 0))
    #
    #             elif s_r + s_g + s_b < 120:
    #
    #                 second_img.putpixel((constant, variable), (s_r2, s_g2, s_b2))
    #
    #
    #         rgb_loss = []
    #         for variable in range(fx, fx * 2):
    #             f_r, f_g, f_b = first_img.getpixel((constant, variable))
    #             s_r, s_g, s_b = second_img.getpixel((constant, variable))
    #
    #             loss = int(((f_r + f_g + f_b) - (s_r + s_g + s_b)) / 3)
    #             rgb_loss.append(loss)
    #
    #         balance = int(sum(rgb_loss) / len(rgb_loss))
    #
    #         for variable in range(balance_range):  # 亮度平衡
    #             f_r, f_g, f_b = first_img.getpixel((constant, variable))
    #             s_r, s_g, s_b = second_img.getpixel((constant, variable))
    #             if (f_r + f_g + f_b > 100):
    #                 f_r = f_r - int((balance / 2) * (balance_range - variable) / balance_range)
    #                 f_g = f_g - int((balance / 2) * (balance_range - variable) / balance_range)
    #                 f_b = f_b - int((balance / 2) * (balance_range - variable) / balance_range)
    #
    #             else:
    #                 pass
    #             if (s_r + s_g + s_b > 100):
    #                 s_r = s_r + int((balance / 2) * (balance_range - variable) / balance_range)
    #                 s_g = s_g + int((balance / 2) * (balance_range - variable) / balance_range)
    #                 s_b = s_b + int((balance / 2) * (balance_range - variable) / balance_range)
    #
    #             else:
    #                 pass
    #
    #             first_img.putpixel((constant, variable), (f_r, f_g, f_b))
    #             second_img.putpixel((constant, variable), (s_r, s_g, s_b))
    #
    #         r0, g0, b0 = first_img.getpixel((constant, 0))
    #         r1, g1, b1 = first_img.getpixel((constant, 1))
    #
    #         r2, g2, b2 = second_img.getpixel((constant, 0))
    #         r3, g3, b3 = second_img.getpixel((constant, 1))
    #
    #         r = int((r0 + r1 + r2 + r3) / 4)
    #         g = int((g0 + g1 + g2 + g3) / 4)
    #         b = int((b0 + b1 + b2 + b3) / 4)
    #
    #         # r = int((r0  + r2 ) / 2)
    #         # g = int((g0  + g2 ) / 2)
    #         # b = int((b0 + b2 ) / 2)
    #
    #         for variable in range(2):  # 均色填充
    #             # pass
    #             first_img.putpixel((constant, variable), (r, g, b))
    #             second_img.putpixel((constant, variable), (r, g, b))
    #
    #
    #
    #         for variable in range(add_light_range):  # 就只是增加亮度
    #             f_r, f_g, f_b = first_img.getpixel((constant, variable))
    #             s_r, s_g, s_b = second_img.getpixel((constant, variable))
    #
    #             if (f_r + f_g + f_b > 100):
    #                 fr = f_r + int((add_light_range - variable) / add_light_range * light) + c
    #                 fg = f_g + int((add_light_range - variable) / add_light_range * light) + c
    #                 fb = f_b + int((add_light_range - variable) / add_light_range * light) + c
    #             else:
    #                 fr = f_r
    #                 fg = f_g
    #                 fb = f_b
    #
    #             if (s_r + s_g + s_b > 100):
    #                 sr = s_r + int((add_light_range - variable) / add_light_range * light) + c
    #                 sg = s_g + int((add_light_range - variable) / add_light_range * light) + c
    #                 sb = s_b + int((add_light_range - variable) / add_light_range * light) + c
    #             else:
    #                 sr = s_r
    #                 sg = s_g
    #                 sb = s_b
    #
    #             first_img.putpixel((constant, variable), (fr, fg, fb))
    #             second_img.putpixel((constant, variable), (sr, sg, sb))
    #
    # if first_img_name=="front.jpg":
    #
    #     for constant in range(1, height - 1):
    #
    #         skip_pixel = 2
    #
    #         # first_img.putpixel((skip_pixel, y), (0, 0, 255))
    #         # second_img.putpixel((width - 1 - skip_pixel, y), (255, 0,0 ))
    #
    #         for variable in range(_range):  # 均色填充
    #
    #             #记住second 是 top
    #
    #             f_r, f_g, f_b = first_img.getpixel((constant, variable))
    #             s_r, s_g, s_b = second_img.getpixel((width-1-constant, width-1-variable))
    #
    #
    #             f_r1, f_g1, f_b1 = first_img_zeros[constant-1,variable, 0], first_img_zeros[constant-1,variable, 1], first_img_zeros[constant-1,variable, 2],
    #             s_r1, s_g1, s_b1 = second_img_zeros[width-1-constant+1, width-1-variable, 0], second_img_zeros[width-1-constant+1, width-1-variable, 1], \
    #                                second_img_zeros[width-1-constant+1, width-1-variable, 2],
    #
    #
    #
    #             try:
    #                 rate_r00 = abs(f_r - f_r1) / f_r
    #                 rate_g00 = abs(f_g - f_g1) / f_g
    #                 rate_b00 = abs(f_b - f_b1) / f_b
    #
    #                 rate_r11 = abs(s_r - s_r1) / s_r
    #                 rate_g11 = abs(s_g - s_g1) / s_g
    #                 rate_b11 = abs(s_b - s_b1) / s_b
    #             except:
    #                 rate_r00, rate_g00, rate_b00 = 2, 2, 2
    #                 rate_r11, rate_g11, rate_b11 = 2, 2, 2
    #
    #             limit_rate00 = (rate_r00 + rate_g00 + rate_b00) / 3
    #             limit_rate11 = (rate_r11 + rate_g11 + rate_b11) / 3
    #
    #
    #             f_r2, f_g2, f_b2 = first_img.getpixel((constant, variable+skip_pixel))
    #             s_r2, s_g2, s_b2 = second_img.getpixel((width-1-constant, width-1-variable-skip_pixel))
    #
    #             r = int((f_r2 + s_r2) / 2 * enhance)  # +   int(light_add*(_range-x)/_range)
    #             g = int((f_g2 + s_g2) / 2 * enhance)  # +   int(light_add*(_range-x)/_range)
    #             b = int((f_b2 + s_b2) / 2 * enhance)  # +  int(light_add*(_range-x)/_range)
    #
    #             if limit_rate00 < threshold1:
    #                 # if  1:
    #                 first_img.putpixel((constant, variable), (r, g, b))
    #
    #                 # left.putpixel((x, y), (0, 0, 255))
    #                 # left.putpixel((x, y), (255, 0, 0))
    #
    #             elif f_r + f_g + f_b < 120:
    #
    #                 #first_img.putpixel((constant, variable), (f_r2, f_g2, f_b2))
    #                 first_img.putpixel((constant, variable), (0, 0, 0))
    #                 #print("这时候xy", (constant, variable))
    #
    #
    #             if limit_rate11 < threshold1:
    #                 # if  1:
    #                 second_img.putpixel((width-1-constant, width-1-variable), (r, g, b))
    #
    #                 # back.putpixel((width - 1 - x, y), ( (0, 0, 255)))
    #                 # back.putpixel((width - 1 - x, y), (255, 0, 0))
    #
    #             elif s_r + s_g + s_b < 120:
    #
    #                 second_img.putpixel((width-1-constant, width-1-variable), (s_r2, s_g2, s_b2))
    #
    #
    #         rgb_loss = []
    #         for variable in range(fx, fx * 2):
    #             f_r, f_g, f_b = first_img.getpixel((constant, variable))
    #             s_r, s_g, s_b = second_img.getpixel((width-1-constant, width-1-variable))
    #
    #             loss = int(((f_r + f_g + f_b) - (s_r + s_g + s_b)) / 3)
    #             rgb_loss.append(loss)
    #
    #         balance = int(sum(rgb_loss) / len(rgb_loss))
    #
    #         for variable in range(balance_range):  # 亮度平衡
    #             f_r, f_g, f_b = first_img.getpixel((constant, variable))
    #             s_r, s_g, s_b = second_img.getpixel((width-1-constant, width-1-variable))
    #             if (f_r + f_g + f_b > 100):
    #                 f_r = f_r - int((balance / 2) * (balance_range - variable) / balance_range)
    #                 f_g = f_g - int((balance / 2) * (balance_range - variable) / balance_range)
    #                 f_b = f_b - int((balance / 2) * (balance_range - variable) / balance_range)
    #
    #             else:
    #                 pass
    #             if (s_r + s_g + s_b > 100):
    #                 s_r = s_r + int((balance / 2) * (balance_range - variable) / balance_range)
    #                 s_g = s_g + int((balance / 2) * (balance_range - variable) / balance_range)
    #                 s_b = s_b + int((balance / 2) * (balance_range - variable) / balance_range)
    #
    #             else:
    #                 pass
    #
    #             first_img.putpixel((constant, variable), (f_r, f_g, f_b))
    #             second_img.putpixel((width-1-constant, width-1-variable), (s_r, s_g, s_b))
    #
    #         r0, g0, b0 = first_img.getpixel((constant, 0))
    #         r1, g1, b1 = first_img.getpixel((constant, 1))
    #
    #         r2, g2, b2 = second_img.getpixel((width-1-constant, 0))
    #         r3, g3, b3 = second_img.getpixel((width-1-constant, 1))
    #
    #         r = int((r0 + r1 + r2 + r3) / 4)
    #         g = int((g0 + g1 + g2 + g3) / 4)
    #         b = int((b0 + b1 + b2 + b3) / 4)
    #
    #         # r = int((r0  + r2 ) / 2)
    #         # g = int((g0  + g2 ) / 2)
    #         # b = int((b0 + b2 ) / 2)
    #
    #         for variable in range(2):  # 均色填充
    #             # pass
    #             first_img.putpixel((constant, variable), (r, g, b))
    #             second_img.putpixel((width-1-constant, width-1-variable), (r, g, b))
    #
    #
    #
    #         for variable in range(add_light_range):  # 就只是增加亮度
    #             f_r, f_g, f_b = first_img.getpixel((constant, variable))
    #             s_r, s_g, s_b = second_img.getpixel((width-1-constant, width-1-variable))
    #
    #             if (f_r + f_g + f_b > 100):
    #                 fr = f_r + int((add_light_range - variable) / add_light_range * light) + c
    #                 fg = f_g + int((add_light_range - variable) / add_light_range * light) + c
    #                 fb = f_b + int((add_light_range - variable) / add_light_range * light) + c
    #             else:
    #                 fr = f_r
    #                 fg = f_g
    #                 fb = f_b
    #
    #             if (s_r + s_g + s_b > 100):
    #                 sr = s_r + int((add_light_range - variable) / add_light_range * light) + c
    #                 sg = s_g + int((add_light_range - variable) / add_light_range * light) + c
    #                 sb = s_b + int((add_light_range - variable) / add_light_range * light) + c
    #             else:
    #                 sr = s_r
    #                 sg = s_g
    #                 sb = s_b
    #
    #             first_img.putpixel((constant, variable), (fr, fg, fb))
    #             second_img.putpixel((width-1-constant, width-1-variable), (sr, sg, sb))

    if first_img_name=="back.jpg":

        for constant in range(1, height - 1):

            skip_pixel = 2

            # first_img.putpixel((skip_pixel, y), (0, 0, 255))
            # second_img.putpixel((width - 1 - skip_pixel, y), (255, 0,0 ))

            for variable in range(_range):  # 均色填充

                #记住second 是 top

                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((width-1-constant, variable))


                f_r1, f_g1, f_b1 = first_img_zeros[constant-1,variable, 0], first_img_zeros[constant-1,variable, 1], first_img_zeros[constant-1,variable, 2],
                s_r1, s_g1, s_b1 = second_img_zeros[width-1-constant+1,variable, 0], second_img_zeros[width-1-constant+1,variable, 1], \
                                   second_img_zeros[width-1-constant+1,variable, 2],



                try:
                    rate_r00 = abs(f_r - f_r1) / f_r
                    rate_g00 = abs(f_g - f_g1) / f_g
                    rate_b00 = abs(f_b - f_b1) / f_b

                    rate_r11 = abs(s_r - s_r1) / s_r
                    rate_g11 = abs(s_g - s_g1) / s_g
                    rate_b11 = abs(s_b - s_b1) / s_b
                except:
                    rate_r00, rate_g00, rate_b00 = 2, 2, 2
                    rate_r11, rate_g11, rate_b11 = 2, 2, 2

                limit_rate00 = (rate_r00 + rate_g00 + rate_b00) / 3
                limit_rate11 = (rate_r11 + rate_g11 + rate_b11) / 3


                f_r2, f_g2, f_b2 = first_img.getpixel((constant, variable+skip_pixel))
                s_r2, s_g2, s_b2 = second_img.getpixel((width-1-constant, variable+skip_pixel))

                r = int((f_r2 + s_r2) / 2 * enhance)  # +   int(light_add*(_range-x)/_range)
                g = int((f_g2 + s_g2) / 2 * enhance)  # +   int(light_add*(_range-x)/_range)
                b = int((f_b2 + s_b2) / 2 * enhance)  # +  int(light_add*(_range-x)/_range)

                if limit_rate00 < threshold1:
                    # if  1:
                    first_img.putpixel((constant, variable), (r, g, b))

                    # left.putpixel((x, y), (0, 0, 255))
                    # left.putpixel((x, y), (255, 0, 0))

                elif f_r + f_g + f_b < 120:

                    first_img.putpixel((constant, variable), (f_r2, f_g2, f_b2))
                    #first_img.putpixel((constant, variable), (0, 0, 0))
                    #print("这时候xy", (constant, variable))


                if limit_rate11 < threshold1:
                    # if  1:
                    second_img.putpixel((width-1-constant, variable), (r, g, b))

                    # back.putpixel((width - 1 - x, y), ( (0, 0, 255)))
                    # back.putpixel((width - 1 - x, y), (255, 0, 0))

                elif s_r + s_g + s_b < 120:

                    second_img.putpixel((width-1-constant, variable), (s_r2, s_g2, s_b2))


            rgb_loss = []
            for variable in range(fx, fx * 2):
                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((width-1-constant, variable))

                loss = int(((f_r + f_g + f_b) - (s_r + s_g + s_b)) / 3)
                rgb_loss.append(loss)

            balance = int(sum(rgb_loss) / len(rgb_loss))

            for variable in range(balance_range):  # 亮度平衡
                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((width-1-constant, variable))
                if (f_r + f_g + f_b > 100):
                    f_r = f_r - int((balance / 2) * (balance_range - variable) / balance_range)
                    f_g = f_g - int((balance / 2) * (balance_range - variable) / balance_range)
                    f_b = f_b - int((balance / 2) * (balance_range - variable) / balance_range)

                else:
                    pass
                if (s_r + s_g + s_b > 100):
                    s_r = s_r + int((balance / 2) * (balance_range - variable) / balance_range)
                    s_g = s_g + int((balance / 2) * (balance_range - variable) / balance_range)
                    s_b = s_b + int((balance / 2) * (balance_range - variable) / balance_range)

                else:
                    pass

                first_img.putpixel((constant, variable), (f_r, f_g, f_b))
                second_img.putpixel((width-1-constant, variable), (s_r, s_g, s_b))

            r0, g0, b0 = first_img.getpixel((constant, 0))
            r1, g1, b1 = first_img.getpixel((constant, 1))

            r2, g2, b2 = second_img.getpixel((width-1-constant, 0))
            r3, g3, b3 = second_img.getpixel((width-1-constant, 1))

            r = int((r0 + r1 + r2 + r3) / 4)
            g = int((g0 + g1 + g2 + g3) / 4)
            b = int((b0 + b1 + b2 + b3) / 4)

            # r = int((r0  + r2 ) / 2)
            # g = int((g0  + g2 ) / 2)
            # b = int((b0 + b2 ) / 2)

            for variable in range(2):  # 均色填充
                # pass
                first_img.putpixel((constant, variable), (r, g, b))
                second_img.putpixel((width-1-constant, variable), (r, g, b))



            for variable in range(add_light_range):  # 就只是增加亮度
                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((width-1-constant, variable))

                if (f_r + f_g + f_b > 100):
                    fr = f_r + int((add_light_range - variable) / add_light_range * light) + c
                    fg = f_g + int((add_light_range - variable) / add_light_range * light) + c
                    fb = f_b + int((add_light_range - variable) / add_light_range * light) + c
                else:
                    fr = f_r
                    fg = f_g
                    fb = f_b

                if (s_r + s_g + s_b > 100):
                    sr = s_r + int((add_light_range - variable) / add_light_range * light) + c
                    sg = s_g + int((add_light_range - variable) / add_light_range * light) + c
                    sb = s_b + int((add_light_range - variable) / add_light_range * light) + c
                else:
                    sr = s_r
                    sg = s_g
                    sb = s_b

                first_img.putpixel((constant, variable), (fr, fg, fb))
                second_img.putpixel((width-1-constant, variable), (sr, sg, sb))

    if first_img_name=="front.jpg":

        for constant in range(1, height - 1):

            skip_pixel = 2

            # first_img.putpixel((skip_pixel, y), (0, 0, 255))
            # second_img.putpixel((width - 1 - skip_pixel, y), (255, 0,0 ))

            for variable in range(_range):  # 均色填充

                #记住second 是 top

                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((constant, width-1-variable))


                f_r1, f_g1, f_b1 = first_img_zeros[constant-1,variable, 0], first_img_zeros[constant-1,variable, 1], first_img_zeros[constant-1,variable, 2],
                s_r1, s_g1, s_b1 = second_img_zeros[constant-1, width-1-variable, 0], second_img_zeros[constant-1, width-1-variable, 1], \
                                   second_img_zeros[constant-1, width-1-variable, 2],



                try:
                    rate_r00 = abs(f_r - f_r1) / f_r
                    rate_g00 = abs(f_g - f_g1) / f_g
                    rate_b00 = abs(f_b - f_b1) / f_b

                    rate_r11 = abs(s_r - s_r1) / s_r
                    rate_g11 = abs(s_g - s_g1) / s_g
                    rate_b11 = abs(s_b - s_b1) / s_b
                except:
                    rate_r00, rate_g00, rate_b00 = 2, 2, 2
                    rate_r11, rate_g11, rate_b11 = 2, 2, 2

                limit_rate00 = (rate_r00 + rate_g00 + rate_b00) / 3
                limit_rate11 = (rate_r11 + rate_g11 + rate_b11) / 3


                f_r2, f_g2, f_b2 = first_img.getpixel((constant, variable+skip_pixel))
                s_r2, s_g2, s_b2 = second_img.getpixel((constant, width-1-variable-skip_pixel))

                r = int((f_r2 + s_r2) / 2 * enhance)  # +   int(light_add*(_range-x)/_range)
                g = int((f_g2 + s_g2) / 2 * enhance)  # +   int(light_add*(_range-x)/_range)
                b = int((f_b2 + s_b2) / 2 * enhance)  # +  int(light_add*(_range-x)/_range)

                if limit_rate00 < threshold1:
                    # if  1:
                    first_img.putpixel((constant, variable), (r, g, b))

                    # left.putpixel((x, y), (0, 0, 255))
                    # left.putpixel((x, y), (255, 0, 0))

                elif f_r + f_g + f_b < 120:

                    first_img.putpixel((constant, variable), (f_r2, f_g2, f_b2))
                    #first_img.putpixel((constant, variable), (0, 0, 0))
                    #print("这时候xy", (constant, variable))


                if limit_rate11 < threshold1:
                    # if  1:
                    second_img.putpixel((constant, width-1-variable), (r, g, b))

                    # back.putpixel((width - 1 - x, y), ( (0, 0, 255)))
                    # back.putpixel((width - 1 - x, y), (255, 0, 0))

                elif s_r + s_g + s_b < 120:

                    second_img.putpixel((constant, width-1-variable), (s_r2, s_g2, s_b2))


            rgb_loss = []
            for variable in range(fx, fx * 2):
                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((constant, width-1-variable))

                loss = int(((f_r + f_g + f_b) - (s_r + s_g + s_b)) / 3)
                rgb_loss.append(loss)

            balance = int(sum(rgb_loss) / len(rgb_loss))

            for variable in range(balance_range):  # 亮度平衡
                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((constant, width-1-variable))
                if (f_r + f_g + f_b > 100):
                    f_r = f_r - int((balance / 2) * (balance_range - variable) / balance_range)
                    f_g = f_g - int((balance / 2) * (balance_range - variable) / balance_range)
                    f_b = f_b - int((balance / 2) * (balance_range - variable) / balance_range)

                else:
                    pass
                if (s_r + s_g + s_b > 100):
                    s_r = s_r + int((balance / 2) * (balance_range - variable) / balance_range)
                    s_g = s_g + int((balance / 2) * (balance_range - variable) / balance_range)
                    s_b = s_b + int((balance / 2) * (balance_range - variable) / balance_range)

                else:
                    pass

                first_img.putpixel((constant, variable), (f_r, f_g, f_b))
                second_img.putpixel((constant, width-1-variable), (s_r, s_g, s_b))

            r0, g0, b0 = first_img.getpixel((constant, 0))
            r1, g1, b1 = first_img.getpixel((constant, 1))

            r2, g2, b2 = second_img.getpixel((constant, width-1-0))
            r3, g3, b3 = second_img.getpixel((constant, width-1-1))

            r = int((r0 + r1 + r2 + r3) / 4)
            g = int((g0 + g1 + g2 + g3) / 4)
            b = int((b0 + b1 + b2 + b3) / 4)

            # r = int((r0  + r2 ) / 2)
            # g = int((g0  + g2 ) / 2)
            # b = int((b0 + b2 ) / 2)

            for variable in range(2):  # 均色填充

                first_img.putpixel((constant, variable), (r, g, b))
                second_img.putpixel((constant, width-1-variable), (r, g, b))

                # first_img.putpixel((constant, variable), (0, 0, 0))
                # second_img.putpixel((constant, width-1-variable), (0, 0, 0))



            for variable in range(add_light_range):  # 就只是增加亮度
                f_r, f_g, f_b = first_img.getpixel((constant, variable))
                s_r, s_g, s_b = second_img.getpixel((constant, width-1-variable))

                if (f_r + f_g + f_b > 100):
                    fr = f_r + int((add_light_range - variable) / add_light_range * light) + c
                    fg = f_g + int((add_light_range - variable) / add_light_range * light) + c
                    fb = f_b + int((add_light_range - variable) / add_light_range * light) + c
                else:
                    fr = f_r
                    fg = f_g
                    fb = f_b

                if (s_r + s_g + s_b > 100):
                    sr = s_r + int((add_light_range - variable) / add_light_range * light) + c
                    sg = s_g + int((add_light_range - variable) / add_light_range * light) + c
                    sb = s_b + int((add_light_range - variable) / add_light_range * light) + c
                else:
                    sr = s_r
                    sg = s_g
                    sb = s_b

                first_img.putpixel((constant, variable), (fr, fg, fb))
                second_img.putpixel((constant, width-1-variable), (sr, sg, sb))




    if (green_line):
        pass
        # for y in range(height):
        #     first_img.putpixel((fx, y), (0, 255, 0))
        #     second_img.putpixel((width - 1 - fx, y), (0, 255, 0))

    first_img = first_img.convert('RGB')
    second_img = second_img.convert('RGB')

    # first_img.save("2048_processed/left.jpg")
    first_img.save(output_path + "/" + first_img_name)
    second_img.save(output_path + "/" + second_img_name)

    # 使用双边滤波进行处理
    if bilateralFilter:
        first_img = cv2.imread(output_path + "/" + first_img_name)
        second_img = cv2.imread(output_path + "/" + second_img_name)

        first_img = cv2.bilateralFilter(first_img, 9, 75, 75)
        second_img = cv2.bilateralFilter(second_img, 9, 75, 75)

        cv2.imwrite(output_path + "/" + first_img_name, first_img)
        cv2.imwrite(output_path + "/" + second_img_name, second_img)


def Zc_CopyDown(output_path):
    down_Img = Image.open(input_img_path + '/' + "down.jpg")  # 打开图片
    down_Img = down_Img.convert('RGB')
    down_Img.save(output_path+"/"+"down.jpg")

def tif2jpg(name):
    down_Img = Image.open(name)  # 打开图片
    down_Img = down_Img.convert('RGB')
    #down_Img.save(name.replace('.tif','.jpg'))
    down_Img.save(name.replace('_sphere.tif','{}.jpg'.format(sphere_name)))
    #os.remove(name)

#input_path="D:/ZC"
input_path=input_img_path
#output_path=r"D:\Users\dunch\PycharmProjects\pythonProject\2048_processed"

color_line=0
fuzzy=0
bilateralFilter=0


process_top=1
process_horizontal=1

start=time.time()
print("开始处理顶上4条边")
if process_top:
    fx=4

    first_img="front.jpg"
    second_img="top.jpg"

    Zc_PixelPadding_Top2(first_img,second_img,output_img_path,output_img_path,fx,color_line,fuzzy)


    first_img="back.jpg"
    second_img="top.jpg"

    Zc_PixelPadding_Top2(first_img,second_img,output_img_path,output_img_path,fx,color_line,fuzzy)


    first_img="left.jpg"
    second_img="top.jpg"

    Zc_PixelPadding_Top2(first_img,second_img,output_img_path,output_img_path,fx,color_line,fuzzy)


    first_img="right.jpg"
    second_img="top.jpg"

    Zc_PixelPadding_Top2(first_img,second_img,output_img_path,output_img_path,fx,color_line,fuzzy)

#####################################################################################################################
print("开始处理四周4条棱")
if process_horizontal:

    fx=4
    first_img="left.jpg"
    second_img="back.jpg"
    #Zc_PixelPadding_Horizontal(first_img,second_img,input_path,input_path,fx,color_line,fuzzy)
    Zc_PixelPadding_Horizontal(first_img,second_img,output_img_path,output_img_path,fx,color_line,fuzzy)



    first_img="front.jpg"
    second_img="left.jpg"
    Zc_PixelPadding_Horizontal(first_img,second_img,output_img_path,output_img_path,fx,color_line,fuzzy)


    first_img="right.jpg"
    second_img="front.jpg"
    Zc_PixelPadding_Horizontal(first_img, second_img, output_img_path, output_img_path, fx, color_line, fuzzy)


    first_img="back.jpg"
    second_img="right.jpg"
    Zc_PixelPadding_Horizontal(first_img,second_img,output_img_path,output_img_path,fx,color_line,fuzzy)


    ############################################################################################################################

#Zc_CopyDown(output_img_path)

end=time.time()
#new_path=input_path

new_path=output_img_path

f11=new_path+"\\"+"front.jpg"
f22=new_path+"\\"+"back.jpg"
f33=new_path+"\\"+"top.jpg"
f44=new_path+"\\"+"down.jpg"
f55=new_path+"\\"+"left.jpg"
f66=new_path+"\\"+"right.jpg"


# 先访问有没有全景图，有就删除

# sphere_img = output_img_path + "\\" + "_sphere.tif"
#
# print("要删除的文件全路径是：", sphere_img)
# try:
#
#     os.remove(sphere_img)
# except:
#     pass


#sphere_img = output_img_path + "\\" + "_sphere.tif"

path = output_img_path + "\\" + "_sphere.tif"

if os.path.exists(path):  # 如果文件存在
    # 删除文件，可使用以下两种方法。
    os.remove(path)


# # print(new_path+"//"+"_shpere.tif")
# os.system("D:/Krpano/krpano-1.19-pr13/1.bat {} {} {} {} {} {}".format(f11,f22,f33,f44,f55,f66))
# time.sleep(2)
# print(" y ")
# os.system("D:/Krpano/krpano-1.19-pr13/2.bat {}".format(new_path+"//"+"_sphere.tif"))
# time.sleep(2)
#
# print(" y ")
# os.system("{}".format(new_path+"//"+"vtour"+"//"+"tour_testingserver.exe"))


p = subprocess.Popen("{} {} {} {} {} {} {}".format(krpano_path,f11,f22,f33,f44,f55,f66),shell=True)
# p = subprocess.Popen("{} {} {} {} {} {} {}".format(krpano_path,f11,f22,f33,f44,f55,f66))
# time.sleep(5)
# p.kill()





#sphere_img=output_img_path+"\\"+"_sphere.tif"


#print(sphere_img)





while True:
    try:
        sphere_img=output_img_path+"\\"+"_sphere.tif"
        #print("要读取的文件位置：{}".format(sphere_img))

        #p.kill()为什么不能Kill这个进程？使用kill会

        tif2jpg(sphere_img)

        break
    except:
        continue


# 这是用来弹网页的
# print("跳出1.bat循环了")
#
# sphere_tif=output_img_path+"\\"+"_sphere.tif"
# p = subprocess.Popen("D:/Krpano/krpano-1.19-pr13/2.bat {}".format(sphere_tif))
# #time.sleep(2)
#
#
# while True:
#     try:
#         print("asdasdads")
#         p = subprocess.Popen("{}".format(output_img_path + "//" + "vtour" + "//" + "tour_testingserver.exe"))
#         break
#     except:
#         continue




# p.kill()
#
# p = subprocess.Popen("{}".format(new_path+"//"+"vtour"+"//"+"tour_testingserver.exe"))
# time.sleep(3)
# p.kill()



#tif2jpg(sphere_img)


#   PixelPadding.py D:\ZC\PNG D:\OutPut_Test D:/Krpano/krpano-1.19-pr13/1.bat new_name

#p.terminate()
#time.sleep(5)

#exe_name="{}".format(output_img_path + "//" + "vtour" + "//" + "tour_testingserver.exe")
#exe_name=r"C:\WINDOWS\system32\cmd.exe"


print("共使用{}秒".format( round((end-start-1),3)  ))
#time.sleep(1)
#p.kill()


#没有那个弹网页的额外cmd，也不需要关闭了
# time.sleep(2)


'''
exe_name="cmd.exe"
os.system('taskkill /f /t /im '+exe_name)#MESMTPC.exe程序名字

'''




# exe_name="PixelPadding.exe"
# os.system('taskkill /f /t /im '+exe_name)#MESMTPC.exe程序名字



sys.exit()
