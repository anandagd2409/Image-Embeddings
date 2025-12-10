# Image Embeddings With ONNX for Semantic Search 

This project demonstrates how to generate **image embeddings externally** using an ONNX ResNet-50 model and perform **semantic similarity search** inside **Oracle Database 26ai** using its native vector datatype and VECTOR_DISTANCE() function.

The application is built using **Streamlit**, **Python**, and **cx_Oracle**, and supports image upload, embedding generation, vector storage, and similarity-based retrieval.

# Features
##  External Embeddings (Bring Your Own Model)
- Uses ONNX ResNet-50 to generate 2048-dimensional embeddings.
- Runs inference outside Oracle (fully decoupled).
## Oracle 23ai Vector Search
- Stores vectors using Oracle’s native VECTOR datatype.
- Performs COSINE similarity via VECTOR_DISTANCE().
## Full Streamlit App
- Upload images
- Preview images
- Extract embeddings
- Store image + metadata + vector
- Run similarity search
- Display top matches visually

## Tech Stack
- Python 3
- Streamlit (UI)
- ONNX Model – ResNet-50
- MXNet / ONNX Runtime
- Oracle 26ai (Vector datatype + SQL search)
- cx_Oracle for DB connectivity

## Architecture
<img width="712" height="294" alt="image" src="https://github.com/user-attachments/assets/c7dc9152-78ff-4bc9-a93b-e89420a944d6" />

## Table Structure

The app automatically creates:

**STREAMLIT_IMAGES**

- id (identity PK)
- filename
- file_extension
- upload_date
- image_data (BLOB)
- description
- v32 (VECTOR)

**STREAMLIT_STG_IMAGES**
Staging table for searching against existing images.

## Core Code Snippet (Embedding → Storage → Search)

### **Generate embeddings with ONNX:-**
`vector_data = gen_embeddings(img_bytes_io)`
`vector_json = json.dumps(vector_data)`
### **Insert into Oracle:-**
`cursor.execute("""
    INSERT INTO STREAMLIT_IMAGES 
    (filename, file_extension, image_data, description, v32)
    VALUES (:1, :2, :3, :4, TO_VECTOR(:5))
""", (filename, file_extension, blob_var, description, vector_json))`
### **Vector similarity search:-**
`cursor.execute("""
    SELECT id, filename, VECTOR_DISTANCE(v32, TO_VECTOR(:1), COSINE) AS distance
    FROM STREAMLIT_IMAGES
    ORDER BY distance
    FETCH FIRST 5 ROWS ONLY
""", [vector_json])`

## Run the Application
1. Activate Python environment
`source /etc/alternatives/py39env/bin/activate`
  
2. Run Streamlit
`python -m streamlit run upload_img_to_db.py --server.address 0.0.0.0 --server.port 8501`

Then open:
`http://localhost:8501`

Project Structure
