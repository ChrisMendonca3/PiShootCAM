# import necessary libraries
 
import cv2
import numpy as np
import os
webcam = cv2.VideoCapture(0)
while True:
  
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd
        
        cv2.imshow("Capturing", frame)
        
        cv2.imwrite(filename='D:\Chris\SEM 5\pr\saved_img.jpg', img=frame)

        webcam.release()
        cv2.waitKey(0)
        break 
# Turn on Laptop's webcam
frame = cv2.imread('D:\Chris\SEM 5\pr\saved_img.jpg')
 
#while True:
     
#    ret, frame = cap.read()
 
    # Locate points of the documents
    # or object which you want to transform
pts1 = np.float32([[117,110], [2804, 74], [97, 2804], [2849, 2819]])
pts2 = np.float32([[0, 0], [1000, 0], [0, 1000], [1000, 1000]])


    # Apply Perspective Transform Algorithm
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(frame, matrix, (1000, 1000))
     
    # Wrap the transformed image
cv2.imshow('frame', frame) # Initial Capture
cv2.waitKey(0)
cv2.imshow('frame1', result) # Transformed Capture
cv2.waitKey(0)

filename = "c:\\Anand\\Image\\saved_img.jpg"
cv2.imwrite(filename, result)








img = cv2.imread("D:\Chris\SEM 5\pr\saved_img.jpg", cv2.IMREAD_COLOR)

  
# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3, 3))

filename = "c:\\Anand\\Image\\saved_img.jpg"
cv2.imwrite(filename, gray_blurred)

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
               param2 = 30, minRadius = 1, maxRadius = 200)
  
# Draw circles that are detected.
if detected_circles is not None:
  
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    outer_ring = detected_circles[0][0]
    x1,y1,r1 = outer_ring[0],outer_ring[1],outer_ring[2]
    cv2.circle(img, (x1, y1), r1, (0, 255, 0), 2)
    # Draw a small circle (of radius 1) to show the center.
    cv2.circle(img, (x1, y1), 1, (0, 0, 255), 3)
    cv2.imshow("Detected Circle", img)
    cv2.waitKey(0)
 

    pellet_ring = detected_circles[0][-1]
    x2,y2,r2 = pellet_ring[0],pellet_ring[1],pellet_ring[2]
    cv2.circle(img, (x2, y2), r2, (0, 255, 0), 2)
    # Draw a small circle (of radius 1) to show the center.
    cv2.circle(img, (x2, y2), 1, (0, 0, 255), 3)
    cv2.imshow("Detected Circle", img)
    cv2.waitKey(0)
    
    cv2.line(img, (x1,y1),(x2,y2), (0,0,0),2)
    cv2.imshow("Detected Circle", img)
    cv2.waitKey(0)

    distance = ((int(x1)  - int(x2))**2 + (int(y1)  - int(y2))**2) **0.5
    v_score = distance * -0.020183332  + 10.9
    print("The distance between two points is ", distance)
    print("score", v_score)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    v_text = "Length = " + str(distance)  + " Score = " + str(v_score)
    cv2.putText(img, v_text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Detected Circle", img)
    cv2.waitKey(0)
    
#    for pt in detected_circles[0, :]:
#        print(pt)
#        a, b, r = pt[0], pt[1], pt[2]
# 
#        # Draw the circumference of the circle.
#        cv2.circle(img, (a, b), r, (0, 255, 0), 2)
#  
#        # Draw a small circle (of radius 1) to show the center.
#        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
#        cv2.imshow("Detected Circle", img)
#        cv2.waitKey(0)

#filename = "c:\\Anand\\Image\\processedTG5.jpg"
cv2.imwrite(filename, img)


#deletion of saved.jpg
if os.path.exists("D:\Chris\SEM 5\pr\saved_img.jpg"):
    os.remove("D:\Chris\SEM 5\pr\saved_img.jpg")
else:
        print("null")






cv2.destroyAllWindows()