import imutils
import cv2

redLower = (169, 100, 100)
redUpper = (189, 255, 255)

camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width=600)
    blurredImage = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, redLower, redUpper)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            if radius > 250:
                print("stop")
            else:
                if center[0] < 150:
                    print("Left")
                elif center[0] > 450:
                    print("Right")
                elif radius < 250:
                    print("Front")
                else:
                    print("Stop")

    cv2.imshow("From", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
