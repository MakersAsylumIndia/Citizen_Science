import os
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageFont

# Load the image classification model
model = load_model('C:/users/tanuj/desktop/makers asylum/keras_model.h5')

# Load the labels
class_names = [class_name.strip() for class_name in open('C:/users/tanuj/desktop/makers asylum/labels.txt', 'r').readlines()]

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Define the folder path
folder_path = "C:/users/tanuj/desktop/makers asylum/test_set/"

def extraction_exif():

    print(image_list)
    import pandas as pd
    from GPSPhoto import gpsphoto

    data_location = []
    for a in image_list:
    b = gpsphoto.getGPSData(os.getcwd() + f'\\{a}')
    data_location.append(b)
        
    df = pd.DataFrame(data = data_location, index = image_list)
    print(df.head())
    df.to_csv("C:\\Users\\Praneeth\\Desktop\\GPS\\2nd April 2023\\020423_GPSdata.csv")

    sw = (df.Latitude.max() + 3, df.Longitude.min() - 3)
    ne = (df.Latitude.min() - 3, df.Longitude.max() + 3)

    import folium
    m = folium.Map()

    for lat, lon, in zip(df.Latitude.values, df.Longitude.values,):
        folium.CircleMarker(
            [lat, lon], 
            ).add_to(m)

    m.fit_bounds([sw, ne])
    display(m)


def classify_images_with_ocr(image_folder):
    data = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            path = os.path.join(image_folder, file_name)

            # Read the original image
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize the image for classification
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_resized = img_resized / 255.0
            img_resized = np.expand_dims(img_resized, axis=0)

            # Predict the image class
            prediction = model.predict(img_resized)
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]
            class_name = class_names[predicted_class]

            # Perform OCR on the image
            result = ocr.ocr(img, cls=True)
            ocr_text = [line[1][0] for line in result[0]]

            # Extract result information
            result = result[0]
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]

            # Draw OCR text on the image
            im_show = draw_ocr(img_rgb, boxes, txts, scores, font_path='C:/Users/tanuj/Desktop/Makers ASylum/Times-New-Roman/times new roman.ttf')
            im_show = Image.fromarray(im_show)

            # Save the result image with the updated filename
            output_filename = f"{file_name}_result.jpg"
            output_filepath = os.path.join(image_folder, output_filename)
            im_show.save(output_filepath)

            # Store the image path, predicted class, confidence score, and OCR text in a dictionary
            data.append({
                'Image': path,
                'Class': class_name,
                'Confidence': confidence,
                'OCR Text': ocr_text,
                'Latitude': df.Latitude,
                'Longitude': df.Longitude
            })

    # Create a dataframe from the collected data
    df = pd.DataFrame(data)
    return df

result_df = classify_images_with_ocr(folder_path)

# Print the resulting dataframe
print(result_df)

# Check if the classification_results.csv file exists and has columns
if not os.path.isfile('classification_results.csv') or os.stat('classification_results.csv').st_size == 0:
    # If the file does not exist or is empty, create a new empty CSV file with column headers
    empty_df = pd.DataFrame(columns=['Image', 'Class', 'Confidence', 'OCR Text'])
    empty_df.to_csv('classification_results.csv', index=False)

# Load the existing classification results CSV
classification_results = pd.read_csv('classification_results.csv')

# Merge the OCR results with the classification results based on the 'Image' column
merged_df = pd.merge(classification_results, result_df, on='Image', how='outer')

# Save the merged dataframe to the same CSV file
merged_df.to_csv('classification_results.csv', index=False)
