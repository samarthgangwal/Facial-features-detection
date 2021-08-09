import cv2
import sys

menu = int(input("Enter 1 for face detection using an image \nEnter 2 for face detection using webcam\nEnter Number:"))
    
#Get user supplied values
FcascPath = 'haarcascade face path'
EcascPath = 'haarcascade eyes path'
ScascPath = 'haarcascade smile path'

#Create the haar cascade
faceCascade = cv2.CascadeClassifier(FcascPath)
eyeCascade = cv2.CascadeClassifier(EcascPath)
smileCascade = cv2.CascadeClassifier(ScascPath)

if menu == 1:

    k = input("Enter file name (With Extension):")
    #user values
    imagePath = 'folder path having of file having pictures' + k
    #Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    #Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = image[y:y+h, x:x+w]

    #Detect Eyes
    eyes = eyeCascade.detectMultiScale(roi_gray)

    #Draw a rectangle around eyes
    for (ex,ey,ew,eh) in eyes: 
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
    print("Found {0} eyes!".format(len(eyes)))

    #Detect Smiles
    smiles = smileCascade.detectMultiScale(roi_gray, 1.8, 20)

    #Draw a rectangle around smile
    for (sx, sy, sw, sh) in smiles: 
                cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    print("Found {0} smiles!".format(len(smiles)))

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)

if menu == 2:

    def imgdetect(gray,image):
        #Detect faces in the image
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        

        #Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w] 
            roi_color = image[y:y+h, x:x+w]

            #Detect Eyes
            eyes = eyeCascade.detectMultiScale(roi_gray)
            

            #Draw a rectangle around eyes
            for (ex,ey,ew,eh) in eyes: 
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
                
            #Detect Smiles
            smiles = smileCascade.detectMultiScale(roi_gray, 1.8, 20)
            

            #Draw a rectangle around smile
            for (sx, sy, sw, sh) in smiles: 
                    cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
        return image

    #for video capturing
    vidCapture = cv2.VideoCapture(0)
    while True:
        # To capture video frame by frame
        _, image = vidCapture.read()

        #monochrome image capturing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #calls detect() function
        canvas = imgdetect(gray, image)

        #Displaying Result
        cv2.imshow("Result Video", canvas)

        #To break control by pressing 'q'
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    #Release after processing is done
    vidCapture.release()
    cv2.destroyAllWindows()
        
