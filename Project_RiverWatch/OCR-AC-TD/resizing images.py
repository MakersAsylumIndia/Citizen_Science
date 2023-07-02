import cv2
import os

path = 'C:/users/tanuj/desktop/Makers Asylum/TEACHABLE DATA/DO-METER'
output_path = os.path.join(path, 'resized')

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

for file_name in os.listdir(path):
    if file_name.endswith(".JPG"):
        image_path = os.path.join(path, file_name)
        output_image_path = os.path.join(output_path, file_name)

        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (800, 800))

        # Save the resized image
        cv2.imwrite(output_image_path, resized_image)
        print("Resized:", file_name)
