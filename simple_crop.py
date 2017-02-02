import cv2
read_image = cv2.imread('center_2017_01_29_20_00_19_908.jpg')
crop_img = read_image[60:160, 0:320]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
cv2.imwrite("cropped2.jpg", crop_img)