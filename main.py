import cv2

def print_image():

    img_file='lenna.png'
    img=cv2.imread(img_file, cv2.IMREAD_COLOR)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('lenna_test.png', img)
    
if __name__ == '__main__':
    print_image()

