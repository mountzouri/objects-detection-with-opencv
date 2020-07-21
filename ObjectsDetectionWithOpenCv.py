import cv2
import numpy as np

# Create a VideoCapture object and read from input file
video = cv2.VideoCapture('myVideo.mpeg')

#Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# Compute the number of frames per second, according to the OpenCV version       
if int(major_ver)  < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    #print(fps)
else :
    fps = video.get(cv2.CAP_PROP_FPS)
    #print(fps)

#Default resolutions of the frame are obtained and convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

 
#Create VideoWriter object.The output is stored in 'filtered_video.mpeg' file.
filtered_video = cv2.VideoWriter('filtered_video.mpeg',cv2.VideoWriter_fourcc(*'MPEG'), fps, (frame_width,frame_height))

#Define a function that inserts text to the image, for the Sobel and Canny cases
def insertTextSobel(filter, size, b, g, r, ):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if (size == 3 and filter == 'sobel'):
        cv2.putText(frame, 'Sobel Size = 3', (10,450), font, 0.8, (b, g, r), 2, cv2.LINE_AA)
    elif (size == 5 and filter == 'sobel'):
        cv2.putText(frame, 'Sobel Size = 5', (10,450), font, 0.8, (b, g, r), 2, cv2.LINE_AA)
    elif (size == 50 and filter == 'canny'):
        cv2.putText(frame, 'Canny Thresholds = 50,200', (10,450), font, 0.8, (b, g, r), 2, cv2.LINE_AA)
    elif (size == 150 and filter == 'canny'):
        cv2.putText(frame, 'Canny Thresholds = 150,200', (10,450), font, 0.8, (b, g, r), 2, cv2.LINE_AA)
   

# 5-10sec
# Define Sobel Edge Detector to detect horizontal and vertical edges
def sobelEdgeDetector(img, size):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sobelx = cv2.Sobel (img, cv2.CV_64F,1,0,ksize=size) #apply sobel filter to axis x
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=size) #apply sobel filter to axis y

    #Calculate the absolute values of the horizontal and vertical derivatives
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    
    #Combine the absolute values
    sobel_ = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

    sobel_ = cv2.cvtColor(sobel_, cv2.COLOR_GRAY2BGR) #Converting it, because the video saver expects 3 channels and not 1
    return sobel_

     
    
# 10-15sec
# Define Canny Edge Detector
def cannyEdgeDetector(img, lower):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_ = cv2.Canny(img, lower, 200)  #Apply Canny filter. Two thresholds have to be declared for this reason. One lower and one upper.
    canny_ = cv2.cvtColor(canny_, cv2.COLOR_GRAY2BGR) #Converting it, because the video saver expects 3 channels and not 1
    return canny_


# 15-25sec
# Define a funtion that detects circular shapes
def circleHoughTransform(img):

    # Convert the image to gray, because otherwise cv2.HoughCircles() does not work
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Reduce the noise, using a Median blur, to avoid false circle detection
    gray = cv2.medianBlur(gray, 5)
    
    # Get the number of rows existing in the array/image/frame
    rows = gray.shape[0]
    
    # Detect circles using Hough Transform. Present Three different scenarios
    if (counter < 550):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=0, maxRadius=30)
    elif (counter < 670):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.4, rows / 8, param1=100, param2=20, minRadius=0, maxRadius=6)
    else:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.6, rows / 8, param1=100, param2=30, minRadius=0, maxRadius=30)
   
    # ensure at least some circles were found
    if circles is not None:
        
        circles = np.uint16(np.around(circles))
                
        for i in circles[0, :]:
            center = (i[0], i[1]) #center coordinates
            
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 3) # circle outline
            
    
    return img
 


#Initialize variables for SIFT feature descriptor method    
# Read the sample image
sample_image=cv2.imread("sample.png",0)

# Initialize feature extractor
sift = cv2.xfeatures2d.SIFT_create()
#Use SIFT feature extractor to detect features of the sample image
keypoints_sample, descriptions_sample = sift.detectAndCompute(sample_image,None)


# Initialize feature matcher.
indexParams = dict(algorithm=0,trees=5)
searchParams = dict()
mathcer = cv2.FlannBasedMatcher(indexParams,searchParams)

# 25 -60 sec
# Define SIFT feature descriptor method
def siftDetector(img):
    
    #Convert each frame to gray scale format 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #Use SIFT feature extractor to detect features of each given frame
    keypoints_frame, descriptions_frame = sift.detectAndCompute(gray,None)
    
    #The FLANN based matcher uses knn to match the features with k=2 (2 neighbors)
    knn_matches = mathcer.knnMatch(descriptions_frame, descriptions_sample,k=2)

    
    # Check distance from the most neatest neighbor k and the next neatest neighbor l. 
    # If the distance from point k is less the 70% of the distance on point l, keep k.
    filter_matches=[]    
    for k,l in knn_matches:
        if(k.distance < 0.70*l.distance):
            filter_matches.append(k)


    # Check if the number of matched features is greater than a threshold. 
    # Accept as matches only these cases
    if(len(filter_matches) > 20): #Define a threshold of 20
                
        coordinates_sample = []
        coordinates_frame = []
        
        # Extract the locations (coordinates) of matched keypoints in both the images.
        for k in filter_matches:
            coordinates_sample.append(keypoints_sample[k.trainIdx].pt)
            coordinates_frame.append(keypoints_frame[k.queryIdx].pt)
        
        # Convert te coordinates to numpy lists
        coordinates_sample, coordinates_frame = np.float32((coordinates_sample, coordinates_frame))
        
        # Find transformation constant H, to transform the corners of sample image to corresponding points in each frame
        H,status = cv2.findHomography(coordinates_sample, coordinates_frame, cv2.RANSAC, 3.0)
        
        # Get height and width of the sample image
        height_sample, width_sample = sample_image.shape
        
        # Define border(get the coordinates of the border corners) in sample image
        sample_border = np.float32([[[0,0],[0,height_sample -1],[width_sample -1,height_sample -1],[width_sample -1,0]]])
        
        # Transform the corners of sample image to corresponding points in each frame using H
        frame_border = cv2.perspectiveTransform(sample_border, H)
        
        #Draw the rectangle/border in the current frame 
        cv2.polylines(frame,[np.int32(frame_border)],True,255,5)
              
    
    return frame


#Initialiaze a counter to count the frames        
counter =1
        
# Read until video is completed
while(video.isOpened()):
 
  ret, frame = video.read()
  
  if ret==True:  
    
    if counter < 150.0:                                         #0-5sec           
        frame = frame
    elif counter < 300.0:                                       #5-10sec
        if counter < 240.0:
            frame = sobelEdgeDetector(frame, 3)
            insertTextSobel('sobel', 3, 51, 153, 255)
        else:                
            frame = sobelEdgeDetector(frame, 5)
            insertTextSobel('sobel', 5, 51, 153, 255)
    elif counter < 450.0:                                       #10-15sec
        if counter < 390.0:
            frame = cannyEdgeDetector(frame, 100)
            insertTextSobel('canny', 50, 255, 153, 51)
        else:                
            frame = cannyEdgeDetector(frame, 150)
            insertTextSobel('canny', 150, 255, 153, 51)                                  
    elif counter < 750.0:                                       #15-25sec
        frame = circleHoughTransform(frame)
    elif counter < 900.0:                                       #25-30sec                            
        frame = siftDetector(frame)
    elif counter < 1050.0:                                      #30-35sec
        frame = siftDetector(frame)
    elif counter < 1350.0:                                      #35-45sec
        frame = siftDetector(frame)
    else:                                                       #45-60sec
        frame=siftDetector(frame)
                

    
    #Display the processed video
    cv2.imshow('Frame',frame)
    
    #Write the processed video
    filtered_video.write(frame) 
    
    
    # Press the letter q on keyboard to exit if you wish
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    counter += 1
  
    # Break the loop
  else: 
    break


filtered_video.release()
video.release() 
cv2.destroyAllWindows()
