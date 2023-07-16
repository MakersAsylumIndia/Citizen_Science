import os
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageFont
from GPSPhoto import gpsphoto
import folium
import argparse

# Load the image classification model
model = load_model('./keras_model.h5')

# Load the labels
class_names = [class_name.strip() for class_name in open('./labels.txt', 'r').readlines()]

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Define the folder path containing the images
parser = argparse.ArgumentParser(description='Image classification and OCR script')
parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
args = parser.parse_args()

folder_path = args.folder_path

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

    # Return the dataframe with classification results
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

import os
import pandas as pd
import folium
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif_data(image):
    """Extracts EXIF data from the image file."""
    exif_data = {}
    img = Image.open(image)
    info = img._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            exif_data[decoded] = value
    return exif_data

def get_gps_info(exif_data):
    """Extracts GPS information from the EXIF data."""
    if 'GPSInfo' in exif_data:
        gps_info = {}
        for key in exif_data['GPSInfo'].keys():
            decode = GPSTAGS.get(key, key)
            gps_info[decode] = exif_data['GPSInfo'][key]

        # Converts the GPS latitude and longitude to decimal degrees
        lat = gps_info['GPSLatitude']
        lat_ref = gps_info['GPSLatitudeRef']
        lon = gps_info['GPSLongitude']
        lon_ref = gps_info['GPSLongitudeRef']

        lat_decimal = (lat[0].numerator / lat[0].denominator +
                       lat[1].numerator / lat[1].denominator / 60 +
                       lat[2].numerator / lat[2].denominator / 3600) * (-1 if lat_ref == 'S' else 1)
        lon_decimal = (lon[0].numerator / lon[0].denominator +
                       lon[1].numerator / lon[1].denominator / 60 +
                       lon[2].numerator / lon[2].denominator / 3600) * (-1 if lon_ref == 'W' else 1)

        gps_info['GPSLatitude'] = lat_decimal
        gps_info['GPSLongitude'] = lon_decimal

        return gps_info

image_list = []
for file_name in os.listdir(folder_path):
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        image_path = os.path.join(folder_path, file_name)
        image_list.append(image_path)

lat = []
long = []
img_name = []
for image_path in image_list:
    exif_data = get_exif_data(image_path)
    gps_info = get_gps_info(exif_data)
    if gps_info is not None and 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
        a = gps_info['GPSLatitude']
        b = gps_info['GPSLongitude']
        c = image_path
        lat.append(a)
        long.append(b)
        img_name.append(c)

data = {"Image Name": img_name, "Latitude": lat, "Longitude": long}
df = pd.DataFrame(data)

# Save DataFrame to CSV
csv_file = 'image_coordinates.csv'
df.to_csv(csv_file, index=False)

# Create Folium map
center_lat = df['Latitude'].mean()
center_long = df['Longitude'].mean()
map_osm = folium.Map(location=[center_lat, center_long], zoom_start=12)

# Add markers to the map
for _, row in df.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup=row['Image Name']).add_to(map_osm)

# Save the map to an HTML file
map_file = './static/plotted_map.html'
map_osm.save(map_file)

print("CSV file saved:", csv_file)
print("Map file saved:", map_file)