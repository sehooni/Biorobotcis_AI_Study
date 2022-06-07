# 본 파일은 테스트img와 출력이 나온 img를 동시에 볼수 있도록 plot 시키는 code 입니다.
# 필요한 모듈 import
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


def main():
    # os.getcwd() # 디렉토리 위치 리턴
    # os.chdir('./data/forest_test/')
    file_path = './data/forest_test/'

    # fig = plt.figure()
    rows = 1
    cols = 2
    img_list = []

    for i in range(1, 5):
        fig = plt.figure(i)
        file1_name = file_path + 'test (' + str(i) + ').png'
        file2_name = file_path + 'test (' + str(i) + ').png_forest_type.png'
        img1 = cv2.imread(file1_name, cv2.IMREAD_COLOR)
        img2 = cv2.imread(file2_name, cv2.IMREAD_COLOR)

        ax1 = fig.add_subplot(rows, cols, 1)
        dst1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        ax1.imshow(dst1)
        ax1.set_title('test (' + str(i) + ')')
        ax1.axis("off")

        ax2 = fig.add_subplot(rows, cols, 2)
        dst2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        ax2.imshow(dst2)
        ax2.set_title('pred (' + str(i) + ')')
        ax2.axis("off")

        plt.show()
        cv2.waitKey(0)

if __name__ == '__main__':
    main()

