import time
import cv2 as cv
import os
import numpy as np
# 设置一些常用的一些参数
# 显示的字体 大小 初始位置等
font = cv.FONT_HERSHEY_SIMPLEX #  正常大小无衬线字体
size = 0.5
fx = 10
fy = 355
fh = 18
# ROI框的显示位置
x0 = 300
y0 = 100
# 录制的手势图片大小
width = 300
height = 300
# 每个手势录制的样本数
numofsamples = 300
counter = 0 # 计数器，记录已经录制多少图片了
# 存储地址和初始文件夹名称
gesturename = ''
path = ''
# 标识符 bool类型用来表示某些需要不断变化的状态
binaryMode = False # 是否将ROI显示为而至二值模式
saveImg = False # 是否需要保存图片
lower = np.array([7,28,50])
upper = np.array([20,220,220])
search_model = False
fgbg = cv.createBackgroundSubtractorMOG2()  #创建背景减法背景对象
#鼠标相应函数
# def on_mouse(event, x, y, flags, param):
#     global img, drag_start, patch, point1, point2
#     global w,h
#     img2 = img.copy()
#     if event == cv.EVENT_LBUTTONDOWN:         #左键点击
#         drag_start = True
#        # cv.circle(img2, point1, 5, (0,255,0), 1)
#       #  cv.imshow('image', img2)
#     elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON):               #按住左键拖曳
#         img2 = img.copy()
#         point1 = (x, y)
#         point2 = (point1[0] + 64, point1[1] + 128)
#         cv.rectangle(img2, point1, point2, (255,0,0), 5)
#         cv.imshow('gray', img2)
#     elif event == cv.EVENT_LBUTTONUP:         #左键释放
#         patch = img2[point1[1]:point2[1],point1[0]:point2[0]]   #截取图像
#         drag_start = False
#         cv.imshow('patch',patch)
#         # min_x = min(point1[0],point2[0])
#         # min_y = min(point1[1],point2[1])
#         w = 64
#         h = 128
#         #cut_img = img[min_y:min_y+height, min_x:min_x+width]
#         #cv2.imwrite('lena3.jpg', cut_img)

def binaryMask(frame, x0, y0, width, height):
    #提取ROI像素,
    roi = frame[y0:y0+height, x0:x0+width]
    rows,cols,channels = roi.shape
    image_HSV = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    '''颜色空间转化
    # image_Y = cv.cvtColor(roi, cv.COLOR_BGR2YCR_CB)
    # image_R = cv.cvtColor(roi,cv.COLOR_BGR2RGB)
    #image_HSV = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    '''
#################################以下为图像处理阶段######################################################
    #在此之前可以进行图像分割，得到只含有手势的二值图像
    #获取直方图
    Hist = cv.calcHist([image_HSV],[0,1,2],None,[180,256,256],[0,180,0,256,0,256])
    return cv.normalize(Hist,Hist,0,255,cv.NORM_MINMAX)  #最大值最小值之间进行归一化
    #为了去除噪声，对Cr通道进行高斯滤波
    # Cr1 = cv.GaussianBlur(Cr,(5,5),0)
    # 根据OTSU算法求图像阈值, 对图像进行二值化,先高斯滤波再使用OTSU二值化可以有非常好的滤波效果
    # _, res = cv.threshold(Cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # 保存手势
    """这里可以插入代码调用网络"""

# 保存ROI图像
def saveROI(img):
    global path, counter, gesturename, saveImg
    if counter > numofsamples:
        # 恢复到初始值，以便后面继续录制手势
        saveImg = False
        gesturename = ''
        counter = 0
        return

    counter += 1
    name = gesturename + str(counter) # 给录制的手势命名
    print("Saving img: ", name)
    cv.imwrite(path+name+'.png', img) # 写入文件
    time.sleep(0.05)

# 创建一个视频捕捉对象
cap = cv.VideoCapture(0) # 0为（笔记本）内置摄像头

while(True):
    # 读帧
    ret, frame = cap.read() # 返回的第一个参数为bool类型，用来表示是否读取到帧，如果为False说明已经读到最后一帧。frame为读取到的帧图片
    # 图像翻转（如果没有这一步，视频显示的刚好和我们左右对称）
    frame = cv.flip(frame, 2)# 第二个参数大于0：就表示是沿y轴翻转
    # 显示方框
    cv.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))
    # 显示ROI区域 # 调用函数
    if(binaryMode==True):
        hist = binaryMask(frame, x0, y0, width, height)
        binaryMode = False
        search_model = True
    # 显示提示语
    cv.putText(frame, "Option: ", (fx, fy), font, size, (0, 255, 0))  # 标注字体
    cv.putText(frame, "b-'Binary mode'/ r- 'RGB mode' ", (fx, fy + fh), font, size, (0, 255, 0))  # 标注字体
    cv.putText(frame, "p-'prediction mode'", (fx, fy + 2 * fh), font, size, (0, 255, 0))  # 标注字体
    cv.putText(frame, "s-'new gestures(twice)'", (fx, fy + 3 * fh), font, size, (0, 255, 0))  # 标注字体
    cv.putText(frame, "q-'quit'", (fx, fy + 4 * fh), font, size, (0, 255, 0))  # 标注字体

    key = cv.waitKey(1) & 0xFF # 等待键盘输入，
    if key == ord('b'):  # 将ROI显示为二值模式
       # binaryMode = not binaryMode
       binaryMode = True
       print("Binary Threshold filter active")
    elif key == ord('r'): # RGB模式
        binaryMode = False
    #调节获取图像框的大小   i,j,k,l
    if key == ord('i'):  # 调整ROI框  ？？？
      	y0 = y0 - 5
    elif key == ord('k'):
        y0 = y0 + 5
    elif key == ord('j'):
        x0 = x0 - 5
    elif key == ord('l'):
        x0 = x0 + 5

    if key == ord('p'):
        """调用模型开始预测"""
        print("using CNN to predict")
    if key == ord('q'):
        break

    if key == ord('s'):
        """录制新的手势（训练集）"""
        # saveImg = not saveImg # True
        if gesturename != '':  #
            saveImg = True
        else:
            print("Enter a gesture group name first, by enter press 'n'! ")
            saveImg = False
    elif key == ord('n'):
        # 开始录制新手势
        # 首先输入文件夹名字
        gesturename = (input("enter the gesture folder name: "))
        os.makedirs(gesturename)
        path = "./" + gesturename + "/" # 生成文件夹的地址  用来存放录制的手势
    if search_model == True:
        frame_copy = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([frame_copy],[0,1,2],hist,[0,180,0,256,0,256],1)
        cv.imshow('frame2', dst)
    #展示处理之后的视频帧
    cv.imshow('frame', frame)
    #ROI显示
    cv.imshow("ROI", frame[y0:y0+height, x0:x0+width])

#最后记得释放捕捉
cap.release()
cv.destroyAllWindows()


#工具函数
# 显示ROI为二值模式
# 图像的二值化，就是将图像上的像素点的灰度值设置为0或255，
# 也就是将整个图像呈现出明显的只有黑和白的视觉效果。

#  cv.threshold  进行阈值化
# 第一个参数  src     指原图像，原图像应该是灰度图
# 第二个参数  x     指用来对像素值进行分类的阈值。
# 第三个参数    y  指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
# 有两个返回值 第一个返回值（得到图像的阈值）   二个返回值 也就是阈值处理后的图像

