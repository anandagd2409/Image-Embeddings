# Image Embeddings With ONNX for Semantic Search 

This project demonstrates how to generate **image embeddings externally** using an ONNX ResNet-50 model and perform **semantic similarity search** inside **Oracle Database 26ai** using its native vector datatype and VECTOR_DISTANCE() function.

The application is built using **Streamlit**, **Python**, and **cx_Oracle**, and supports image upload, embedding generation, vector storage, and similarity-based retrieval.

# Features
🔹 External Embeddings (Bring Your Own Model)

Uses ONNX ResNet-50 to generate 2048-dimensional embeddings.

Runs inference outside Oracle (fully decoupled).

🔹 Oracle 23ai Vector Search

    Stores vectors using Oracle’s native VECTOR datatype.
    Performs COSINE similarity via VECTOR_DISTANCE().

🔹 Full Streamlit App

    Upload images
Preview images
Extract embeddings
Store image + metadata + vector
Run similarity search
Display top matches visually
