import numpy as np
import cv2
import math

def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    T, threshInv = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    return threshInv

def adding_fly(path='fly64.png'):
    img = cv2.imread(path)
    h_fly, w_fly, _ = img.shape
    return {'img':img, 'height':h_fly, 'width':w_fly}

def video_processing(file='sample.mp4'):
    WIDTH = 640
    HEIGHT = 480

    #fourcc = cv2.VideoWriter_fourcc(*'XVID')   params of record
    #out = cv2.VideoWriter('capturing.avi', fourcc, 30.0, (640, 480))   start of record

    cap = cv2.VideoCapture(file)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_img = image_processing(frame)
        contours, hierarchy = cv2.findContours(processed_img,
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            dist = math.dist([x+w//2, y+h//2], [WIDTH//2, HEIGHT//2])

            #adding fly to the image
            bound_h1 = y+h//2-adding_fly()['height']//2
            bound_h2 = y+h//2+adding_fly()['height']//2
            bound_w1 = x+w//2-adding_fly()['width']//2
            bound_w2 = x+w//2+adding_fly()['width']//2
            if bound_h1 > 0 and bound_h2 < HEIGHT and bound_w1 > 0 \
                and bound_w2 < WIDTH:           #boundary check
                frame[bound_h1:bound_h2, 
                      bound_w1:bound_w2] = adding_fly()['img']
                
        cv2.putText(frame, f'Dist:{dist:.2f}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (0, 255, 0))
        cv2.imshow('frame', frame)
        #out.write(frame)   writing a record
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    #out.release()  end of record
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_processing()