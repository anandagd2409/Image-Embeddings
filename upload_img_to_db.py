## Purpose: Image Embeddings With ONNX for Semantic Search
## @author: Ananda Ghosh Dastidar
## Version: 1.0

import sys
import numpy as np
np.bool = np.bool_
import mxnet as mx
from collections import namedtuple
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
import array
import onnx
import streamlit as st
import cx_Oracle
from PIL import Image
import io
import os
from datetime import datetime
import json

# Set page config
st.set_page_config(page_title="Image Processing", page_icon="📷", layout="wide")

onnx_model_path = "/home/xxxx/onnx_model/resnet50-v2-7.onnx"
Batch = namedtuple('Batch', ['data'])

# Import the ONNX model
sym, arg_params, aux_params = import_model(onnx_model_path)

# Determine and set context (GPU or CPU)
if len(mx.test_utils.list_gpus()) == 0:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(0)

# Load module for Inferencing
mod = mx.mod.Module(symbol=sym, context=ctx, data_names=['data'], label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))], label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

def get_image(blob_var, show=False):
    # Convert Oracle BLOB to bytes
    blob_data = blob_var.getvalue()
    
    # Create a temporary file-like object from bytes
    img_bytes = io.BytesIO(blob_data)
    
    # Read image using PIL
    img_pil = Image.open(img_bytes)
    
    # Convert PIL image to MXNet image format
    img_mx = mx.nd.array(np.array(img_pil))
    
    if img_mx is None:
        return None
    return img_mx

def preprocess(img):   
    transform_fn = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0)
    return img

def gen_embeddings(blob_var):
    # Convert Oracle BLOB to bytes
    blob_data = blob_var.getvalue()
    
    # Create a temporary file-like object from bytes
    img_bytes = io.BytesIO(blob_data)
    
    # Read image using PIL
    img_pil = Image.open(img_bytes)
    
    # Convert PIL image to MXNet image format
    img_mx = mx.nd.array(np.array(img_pil))
    
    # Preprocess and generate embeddings
    img = preprocess(img_mx)
    mod.forward(Batch([img]))
    scores = mx.ndarray.softmax(mod.get_outputs()[0]).asnumpy()
    
    # Convert to list of floats (Oracle understands this)
    return scores.squeeze().tolist()  # Changed from array.array to list

# Database connection function
def get_db_connection():
    try:
        connection = cx_Oracle.connect(
            user=os.getenv("ORACLE_USER", "xxxxxxxx"),
            password=os.getenv("ORACLE_PASSWORD", "xxxxxxxxxxxx"),
            dsn=os.getenv("ORACLE_DSN", "xxxxx:1521/XXXXXXXXXXXXXXX")
        )
        return connection
    except cx_Oracle.DatabaseError as e:
        st.error(f"Database connection error: {e}")
        return None

# Function to create table if not exists
def create_table_if_not_exists(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT table_name FROM user_tables 
            WHERE table_name = 'STREAMLIT_IMAGES'
        """)
        if not cursor.fetchone():
            cursor.execute("""
                CREATE TABLE STREAMLIT_IMAGES (
                    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    filename VARCHAR2(255),
                    file_extension VARCHAR2(10),
                    upload_date TIMESTAMP DEFAULT SYSTIMESTAMP,
                    image_data BLOB,
                    description VARCHAR2(4000),
                    v32 vector
                )
            """)

        cursor.execute("""
            SELECT table_name FROM user_tables 
            WHERE table_name = 'STREAMLIT_STG_IMAGES'
        """)        
        if not cursor.fetchone():
            cursor.execute("""
                CREATE TABLE streamlit_stg_images (
                    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    filename VARCHAR2(255),
                    file_extension VARCHAR2(10),
                    upload_date TIMESTAMP DEFAULT SYSTIMESTAMP,
                    image_data BLOB,
                    description VARCHAR2(4000),
                    v32 vector
                )
            """)
            connection.commit()         
            st.sidebar.success("Created images table")
    except cx_Oracle.DatabaseError as e:
        st.sidebar.error(f"Table creation error: {e}")

# Function to insert image into database
def insert_image_to_db(filename, file_extension, image_bytes, description):
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            img_bytes_io = io.BytesIO(image_bytes)
            vector_data = gen_embeddings(img_bytes_io)
            
            # Convert to JSON string
            vector_json = json.dumps(vector_data)
            
            blob_var = cursor.var(cx_Oracle.BLOB)
            blob_var.setvalue(0, image_bytes)
            
            cursor.execute("""
                INSERT INTO STREAMLIT_IMAGES 
                (filename, file_extension, image_data, description, v32)
                VALUES (:1, :2, :3, :4, TO_VECTOR(:5))
            """, (filename, file_extension, blob_var, description, vector_json))
            
            connection.commit()
            st.success("✅ Image uploaded successfully!")
            return cursor.rowcount
        except Exception as e:
            st.error(f"Error: {e}")
            return 0
        finally:
            connection.close()
    return 0

def insert_stg_image_to_db(filename, file_extension, image_bytes, description):
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            img_bytes_io = io.BytesIO(image_bytes)
            vector_data = gen_embeddings(img_bytes_io)

            # Convert to JSON string
            vector_json = json.dumps(vector_data)

            # Convert bytes to Oracle BLOB
            blob_var = cursor.var(cx_Oracle.BLOB)
            blob_var.setvalue(0, image_bytes)
            
            # Clear staging table
            cursor.execute("TRUNCATE TABLE STREAMLIT_STG_IMAGES")            
          
            cursor.execute("""
                INSERT INTO STREAMLIT_STG_IMAGES 
                (filename, file_extension, image_data, description, v32)
                VALUES (:1, :2, :3, :4, TO_VECTOR(:5))
            """, (filename, file_extension, blob_var, description, vector_json))
            
            connection.commit()
            st.success("✅ Image uploaded successfully!")
            
            # Perform vector search
            perform_vector_search(connection, vector_data)            
            return cursor.rowcount
        except Exception as e:
            st.error(f"Database insert error: {e}")
            return 0
        finally:
            connection.close()
    return 0

def perform_vector_search(connection, query_vector):
    try:
        cursor = connection.cursor()
        # Convert query vector to JSON
        query_json = json.dumps(query_vector)
        
        cursor.execute("""
            SELECT id, filename, file_extension, upload_date, description,VECTOR_DISTANCE(v32, TO_VECTOR(:1), COSINE) as distance
            FROM STREAMLIT_IMAGES
            ORDER BY distance ASC
            FETCH FIRST 2 ROWS ONLY
        """, [query_json])
        
        results = cursor.fetchall()
        if results:
            st.header("Top 2 Similar Images")
            cols = st.columns(3)
            for idx, row in enumerate(results):
                with cols[idx % 3]:
                    st.write(f"**{row[1]}** (Score: {1 - row[5]:.3f})")
                    image_data = get_image_by_id(row[0])
                    if image_data:
                        display_image(image_data[0], image_data[1])
                        st.caption(f"ID: {row[0]}")
                        if row[4]:  # description
                            st.write(row[4])
        else:
            st.warning("No similar images found.")
            
    except cx_Oracle.DatabaseError as e:
        st.error(f"Vector search error: {e}")

# Function to fetch all images metadata from database
def get_all_images():
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                SELECT id, filename, file_extension, upload_date, description
                FROM STREAMLIT_IMAGES
                ORDER BY upload_date DESC
            """)
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor]
        except cx_Oracle.DatabaseError as e:
            st.error(f"Database query error: {e}")
            return []
        finally:
            connection.close()
    return []

# Function to fetch and read image BLOB data by ID
def get_image_by_id(image_id):
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                SELECT image_data, file_extension, filename
                FROM STREAMLIT_IMAGES
                WHERE id = :id
            """, {'id': image_id})
            
            row = cursor.fetchone()
            if row:
                # Read the LOB data into bytes
                lob_data = row[0].read()
                return (lob_data, row[1], row[2])
            return None
        except cx_Oracle.DatabaseError as e:
            st.error(f"Database query error: {e}")
            return None
        finally:
            connection.close()
    return None

# Function to display image
def display_image(image_data, file_extension):
    try:
        image = Image.open(io.BytesIO(image_data))
        st.image(image, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {e}")

# Main app function
def main():
    st.title("📷 Image Vector Search in Oracle Database 23ai")
    
    # Initialize database table
    conn = get_db_connection()
    if conn:
        create_table_if_not_exists(conn)
        conn.close()
    
    # Sidebar for upload
    with st.sidebar:
        st.header("Upload New Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png", "gif", "bmp"],
            key="uploader"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image preview
            image = Image.open(uploaded_file)
            st.image(image, caption="Upload Preview", use_container_width=True)
            
            # Get file details
            filename = uploaded_file.name
            file_extension = filename.split('.')[-1].lower()
            
            # Add description
            description = st.text_area("Image description (optional)", key="desc")
            
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_bytes = img_byte_arr.getvalue()
            
            # Upload button
            if st.button("Upload to Database"):
                with st.spinner("Uploading image..."):
                    result = insert_image_to_db(
                        filename, 
                        file_extension, 
                        img_bytes, 
                        description
                    )

    with st.sidebar:
        st.header("Upload Image For Vector Search")
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png", "gif", "bmp"],
            key="stg_uploader"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image preview
            image = Image.open(uploaded_file)
            st.image(image, caption="Upload Preview", use_container_width=True)
            
            # Get file details
            filename = uploaded_file.name
            file_extension = filename.split('.')[-1].lower()
            
            # Add description
            #description = st.text_area("Image description (optional)", key="stg_desc")
            description =""
            
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_bytes = img_byte_arr.getvalue()
            
            # Upload button
            if st.button("Search Image"):
                with st.spinner("Uploading image..."):
                    result = insert_stg_image_to_db(
                        filename, 
                        file_extension, 
                        img_bytes, 
                        description
                    )
       
    # Main content area for displaying images
    st.header("Product Catalog in Database")
    
    # Get all images metadata
    images = get_all_images()
    
    if not images:
        st.info("No images found in database. Upload some images using the sidebar.")
    else:
        # Display filter options
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input("Search by filename or description")
        with col2:
            sort_option = st.selectbox("Sort by", ["Newest first", "Oldest first"])
        
        # Filter and sort images
        filtered_images = images
        if search_term:
            filtered_images = [
                img for img in images 
                if (search_term.lower() in img['FILENAME'].lower() or 
                    (img['DESCRIPTION'] and search_term.lower() in img['DESCRIPTION'].lower()))
            ]
        
        if sort_option == "Newest first":
            filtered_images = sorted(filtered_images, key=lambda x: x['UPLOAD_DATE'], reverse=True)
        else:
            filtered_images = sorted(filtered_images, key=lambda x: x['UPLOAD_DATE'])
        
        if not filtered_images:
            st.warning("No images match your search criteria")
        else:
            st.write(f"Showing {len(filtered_images)} image(s)")
            
            # Display images in a grid
            cols = st.columns(6)
            for idx, img in enumerate(filtered_images):
                with cols[idx % 6]:
                    with st.expander(f"{img['FILENAME']} - {img['UPLOAD_DATE'].strftime('%Y-%m-%d %H:%M')}"):
                        # Display image when expanded
                        image_data = get_image_by_id(img['ID'])
                        if image_data:
                            display_image(image_data[0], image_data[1])
                            st.caption(f"ID: {img['ID']}")
                            st.caption(f"Type: {image_data[2].split('.')[-1].upper()}")
                            if img['DESCRIPTION']:
                                st.write(img['DESCRIPTION'])
                            
                            # Download button
                            st.download_button(
                                label="Download Image",
                                data=image_data[0],
                                file_name=image_data[2],
                                mime=f"image/{image_data[1]}",
                                key=f"dl_{img['ID']}"
                            )

if __name__ == "__main__":
    main()