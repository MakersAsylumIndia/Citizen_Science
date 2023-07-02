import os
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageFont
from GPSPhoto import gpsphoto
import folium

# Load the image classification model
model = load_model('C:/users/tanuj/desktop/makers asylum/keras_model.h5')

# Load the labels
class_names = [class_name.strip() for class_name in open('C:/users/tanuj/desktop/makers asylum/labels.txt', 'r').readlines()]

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Define the folder path
folder_path = "C:/users/tanuj/desktop/makers asylum/FINAL_TEST/"

def extraction_exif(image_list):
        data_location = []
        for a in image_list:
            b = gpsphoto.getGPSData(os.path.join(os.getcwd(), a))
            if b is not None and 'Latitude' in b and 'Longitude' in b:
                data_location.append([b['Latitude'], b['Longitude']])

        if data_location:
            df = pd.DataFrame(data=data_location, index=image_list, columns=['Latitude', 'Longitude'])
            df.to_csv("./coordinates.csv")
        else:
            df = pd.DataFrame(columns=['Latitude', 'Longitude'])

        return df

def classify_images_with_ocr(image_folder):
    data = []
    image_list = []
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
            im_show = draw_ocr(img_rgb, boxes, txts, scores, font_path='./times new roman.ttf')
            im_show = Image.fromarray(im_show)

            # Save the result image with the updated filename
            output_filename = f"{file_name}_result.jpg"
            output_filepath = os.path.join(image_folder, output_filename)
            im_show.save(output_filepath)

            image_list.append(file_name)

            # Store the image path, predicted class, confidence score, and OCR text in a dictionary
            data.append({
                'Image': path,
                'Class': class_name,
                'Confidence': confidence,
                'OCR Text': ocr_text
            })

    # Create a dataframe from the collected data
    df = pd.DataFrame(data)

    # Call the extraction_exif function to get GPS coordinates
    gps_df = extraction_exif(image_list)

    # Merge the classification results with the GPS coordinates based on the image filenames
    df = pd.merge(df, gps_df, left_on='Image', right_index=True, how='left')

    # Save the GPS coordinates to a CSV file
    gps_df.to_csv("./coordinates.csv")

    if not gps_df.empty:
        # Create a folium map and plot the coordinates
        sw = (gps_df['Latitude'].max() + 3, gps_df['Longitude'].min() - 3)
        ne = (gps_df['Latitude'].min() - 3, gps_df['Longitude'].max() + 3)
        m = folium.Map()
        for lat, lon in zip(gps_df['Latitude'].values, gps_df['Longitude'].values):
            if not pd.isnull(lat) and not pd.isnull(lon):
                folium.CircleMarker([lat, lon]).add_to(m)
        m.fit_bounds([sw, ne])
        m.save("map.html")

    # Return the dataframe with classification results
    return df

result_df = classify_images_with_ocr(folder_path)

# Print the resulting dataframe
print(result_df)

# Check if the classification_results.csv file exists and has columns
if not os.path.isfile('classification_results.csv') or os.stat('classification_results.csv').st_size == 0:
    # If the file does not exist or is empty, create a new empty CSV file with column headers
    empty_df = pd.DataFrame(columns=['Image', 'Class', 'Confidence', 'OCR Text', 'Latitude', 'Longitude'])
    empty_df.to_csv('classification_results.csv', index=False)

# Load the existing classification results CSV
classification_results = pd.read_csv('classification_results.csv')

# Merge the OCR results with the classification results based on the 'Image' column
merged_df = pd.merge(classification_results, result_df, on='Image', how='outer')

# Save the merged dataframe to the same CSV file
merged_df.to_csv('classification_results.csv', index=False)