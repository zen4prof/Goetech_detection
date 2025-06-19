
import streamlit as st
from PIL import Image as PILImage, ImageDraw
import os
import numpy as np
import onnxruntime
import random

# Assuming feedback_data.py is in the same directory for deployment
try:
    from feedback_data import fault_recommendations
except ImportError:
    st.error("Error loading feedback data. Make sure feedback_data.py is in the same directory.")
    fault_recommendations = {} # Provide an empty dict as fallback


# Set up the Streamlit application title
st.title("Geotechnical Fault Detection Web App (ONNX)")

# Add a file uploader widget
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Path to the exported ONNX model - assuming it's in the same directory as the app.py for deployment
ONNX_MODEL_PATH = 'best.onnx'

@st.cache_resource # Cache the model loading
def load_onnx_model(path):
    if os.path.exists(path):
        try:
            session = onnxruntime.InferenceSession(path, None)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            return session, input_name, output_name
        except Exception as e:
            st.error(f"Error loading ONNX model: {e}")
            return None, None, None
    else:
        st.error(f"ONNX model file not found at {path}")
        return None, None, None

session, input_name, output_name = load_onnx_model(ONNX_MODEL_PATH)

# Class names - Get from data.yaml or training results
CLASS_NAMES = ['Block loss', 'Crack on Ashpat', 'Long Crack', 'Opening on the wall', 'Vegetation on Wall', 'Vegetation on slope', 'Vertical Crack', 'Wall deformation', 'bad foundation', 'corrosion', 'slope deformation']

# Display the uploaded image and run inference
if uploaded_file is not None:
    image = PILImage.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    if session:
        st.write("Running inference using ONNX model...")
        img = np.array(image)
        img_resized = PILImage.fromarray(img).resize((640, 640))
        img_resized = np.array(img_resized)

        img_processed = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img_processed = np.ascontiguousarray(img_processed)
        img_processed = img_processed.astype(np.float32)
        img_processed /= 255.0
        img_processed = np.expand_dims(img_processed, 0)

        try:
            onnx_inputs = {input_name: img_processed}
            onnx_outputs = session.run([output_name], onnx_inputs)
            predictions = onnx_outputs[0].transpose(0, 2, 1)[0]

            confidence_threshold = 0.25

            boxes = predictions[:, :4]
            confidences = np.max(predictions[:, 4:], axis=1)
            class_ids = np.argmax(predictions[:, 4:], axis=1)

            valid_detections = confidences > confidence_threshold

            boxes = boxes[valid_detections]
            class_ids = class_ids[valid_detections]
            confidences = confidences[valid_detections]

            original_width, original_height = image.size
            img_size_model = 640

            boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * (original_width / img_size_model)
            boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * (original_height / img_size_model)
            boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) * (original_width / img_size_model)
            boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) * (original_height / img_size_model)

            draw = ImageDraw.Draw(image)
            colors = {}

            st.write("Inference Results:")
            if len(boxes) > 0:
                for i in range(len(boxes)):
                    box = boxes[i]
                    class_id = int(class_ids[i])
                    confidence = confidences[i]
                    class_name = CLASS_NAMES[class_id]
                    label = f"{class_name}: {confidence:.2f}"

                    if class_id not in colors:
                         colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                    color = colors[class_id]

                    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=2)
                    draw.text((box[0], box[1]), label, fill=color)

                    # Display recommendation
                    if class_name in fault_recommendations:
                        st.write(f"**{class_name}:** {fault_recommendations[class_name]}")
                    else:
                        st.write(f"**{class_name}:** No specific recommendation available.")

                st.image(image, caption="Image with Detections.", use_column_width=True)
            else:
                st.write("No faults detected with the current confidence threshold.")
                st.image(image, caption="Image.", use_column_width=True)


        except Exception as e:
            st.error(f"Error during ONNX inference or postprocessing: {e}")

    else:
        st.warning("ONNX Model could not be loaded, cannot perform inference.")


# Add a footer or additional information
st.write("---")
st.write("Powered by YOLOv8 (ONNX)")
