import argparse
import cv2
import csv

# to store the points for region of interest
roi_pt = []

# to indicate if the left mouse button is depressed
is_button_down = False

def draw_rectangle(event, x, y, flags, param):
    global roi_pt, is_button_down

    if event == cv2.EVENT_MOUSEMOVE and is_button_down:
        global image_clone, image

        # get the original image to paint the new rectangle
        image = image_clone.copy()

        # draw new rectangle
        cv2.rectangle(image, roi_pt[0], (x, y), (0, 255, 0), 2)

    if event == cv2.EVENT_LBUTTONDOWN:
        # record the first point
        roi_pt = [(x, y)]
        is_button_down = True

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        roi_pt.append((x, y))  # append the end point

        # ======================
        # print the bounding box
        # ======================
        # in (x1,y1,x2,y2) format
        print(roi_pt)

        # in (x,y,w,h) format
        bbox = (roi_pt[0][0],
                roi_pt[0][1],
                roi_pt[1][0] - roi_pt[0][0],
                roi_pt[1][1] - roi_pt[0][1])
        print(bbox)

        # button has now been released
        is_button_down = False

        # draw the bounding box
        cv2.rectangle(image, roi_pt[0], roi_pt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

# reference to the image
image_clone = image

# setup the mouse click handler
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_rectangle)

# loop until the 'q' key is pressed
while True:
    # display the image
    cv2.imshow("image", image)

    # wait for a keypress
    key = cv2.waitKey(1)
    if key == ord("c"):
        break

# close all open windows
cv2.destroyAllWindows()