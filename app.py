import streamlit as st
from PIL import Image as PILImage, ImageDraw
import os
import numpy as np
import onnxruntime
import random

# Assuming feedback_data.py is in the same directory for deployment
try:
    # Add the directory of the current script to the Python path
    # This helps in finding feedback_data.py when running the Streamlit app
    import sys
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    from feedback_data import fault_recommendations
except ImportError:
    st.error("Error loading feedback data. Make sure feedback_data.py is in the same directory.")
    fault_recommendations = {} # Provide an empty dict as fallback
except Exception as e:
    st.error(f"An unexpected error occurred loading feedback data: {e}")
    fault_recommendations = {}


# Set up the Streamlit application title
st.title("Geotechnical Fault Detection Web App (ONNX)")

# Add a file uploader widget
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Path to the exported ONNX model - assuming it's in the same directory as the app.py for deployment
ONNX_MODEL_PATH = 'best.onnx'

@st.cache_resource # Cache the model loading
def load_onnx_model(path):
    """Loads the ONNX model and gets input/output names."""
    if not os.path.exists(path):
        st.error(f"ONNX model file not found at {path}")
        return None, None, None
    try:
        # Use the default CPU provider for inference
        session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        return session, input_name, output_name
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        return None, None, None

session, input_name, output_name = load_onnx_model(ONNX_MODEL_PATH)

# Class names - Get from data.yaml or training results
# Make sure these match the order in your trained model
CLASS_NAMES = ['Block loss', 'Crack on Ashpat', 'Long Crack', 'Opening on the wall', 'Vegetation on Wall', 'Vegetation on slope', 'Vertical Crack', 'Wall deformation', 'bad foundation', 'corrosion', 'slope deformation']


def preprocess_image(image: PILImage.Image, target_size=(640, 640)):
    """Preprocesses the image for ONNX model inference."""
    # Resize image while maintaining aspect ratio
    image = image.resize(target_size)
    # Convert to numpy array and normalize
    img_np = np.array(image).astype(np.float32) / 255.0
    # Transpose to (C, H, W)
    img_np = img_np.transpose(2, 0, 1)
    # Add batch dimension
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def postprocess_output(output, img_shape, original_img_shape, confidence_threshold=0.5, iou_threshold=0.45):
    """Processes the raw ONNX model output to get bounding boxes, scores, and class IDs."""
    # The output format depends on the specific YOLOv8 ONNX export.
    # This is a common format: [batch_id, x_center, y_center, width, height, confidence, class1_score, class2_score, ...]
    # Or [batch_id, x1, y1, x2, y2, confidence, class1_score, class2_score, ...]
    # Or [x1, y1, x2, y2, confidence, class_id] for batched outputs

    # Let's assume the output is in the format [batch_id, x1, y1, x2, y2, confidence, class_id] or similar
    # You might need to adjust this based on your actual ONNX model output structure.

    detections = []
    # Assuming output shape is (1, N, 6) where N is number of detections, 6 is [x1, y1, x2, y2, confidence, class_id]
    # Or (1, 6, N) depending on the export settings. Need to inspect the model output shape.

    # A more general approach for YOLOv8 ONNX output (often [1, 84, N] where 84 = 4 box coords + confidence + 80 class scores)
    # Let's assume the output is transposed and in the format [1, N, 84]
    output = np.transpose(output[0], (1, 0)) # Transpose to [N, 84]

    # Filter by confidence
    confidences = output[:, 4] # Confidence score is usually the 5th element
    valid_detections = output[confidences > confidence_threshold]

    boxes = valid_detections[:, :4] # Bounding box coordinates (x1, y1, x2, y2)
    scores = valid_detections[:, 4] # Confidence scores
    class_scores = valid_detections[:, 5:] # Class scores

    # Get class IDs by finding the index of the maximum class score
    class_ids = np.argmax(class_scores, axis=1)

    # Apply Non-Maximum Suppression (NMS)
    # This requires a separate function or library (like torchvision.ops.nms or a custom implementation)
    # For simplicity, let's skip NMS for now and just return the filtered detections.
    # If you need NMS, you'll need to add that logic here.

    # Rescale bounding boxes to original image dimensions
    img_height, img_width = img_shape[0], img_shape[1]
    original_height, original_width = original_img_shape[0], original_img_shape[1]

    # Calculate scaling factors
    x_scale = original_width / img_width
    y_scale = original_height / img_height

    # Rescale boxes
    rescaled_boxes = boxes.copy()
    rescaled_boxes[:, 0] *= x_scale # x1
    rescaled_boxes[:, 1] *= y_scale # y1
    rescaled_boxes[:, 2] *= x_scale # x2
    rescaled_boxes[:, 3] *= y_scale # y2


    for i in range(len(valid_detections)):
        detections.append({
            'box': rescaled_boxes[i].tolist(), # [x1, y1, x2, y2]
            'confidence': scores[i].item(),
            'class_id': class_ids[i].item(),
            'class_name': CLASS_NAMES[class_ids[i].item()] # Map class ID to name
        })

    return detections


# Display the uploaded image and run inference
if uploaded_file is not None and session:
    image = PILImage.open(uploaded_file).convert('RGB')
    original_img_shape = image.size # (width, height)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    if st.button("Detect Faults"):
        st.write("Running inference using ONNX model...")

        # Preprocess the image
        input_image = preprocess_image(image)

        # Run inference
        try:
            onnx_output = session.run([output_name], {input_name: input_image})
            # onnx_output is a list, the first element is the output tensor
            raw_output = onnx_output[0]

            # Postprocess the output
            detections = postprocess_output(raw_output, input_image.shape[2:], original_img_shape)

            st.subheader("Detection Results:")

            if detections:
                # Draw bounding boxes on the image
                draw = ImageDraw.Draw(image)
                for det in detections:
                    box = det['box']
                    label = det['class_name']
                    confidence = det['confidence']
                    # Draw rectangle
                    draw.rectangle(box, outline="red", width=2)
                    # Draw label background
                    text = f"{label}: {confidence:.2f}"
                    try:
                        text_bbox = draw.textbbox((box[0], box[1]), text)
                        draw.rectangle(text_bbox, fill="red")
                        # Draw text
                        draw.text((box[0], box[1]), text, fill="white")
                    except Exception as e:
                         st.warning(f"Could not draw text for detection {label}: {e}")


                    st.write(f"Detected: {label} with confidence {confidence:.2f}")
                    st.write(f"Bounding Box: {box}")

                    # Display feedback based on detected fault
                    if label in fault_recommendations:
                        feedback_info = fault_recommendations[label]
                        st.subheader(f"Feedback for {label}:")
                        st.write(f"Score: {feedback_info.get('score', 'N/A')}")
                        st.write(f"Severity: {feedback_info.get('severity', 'N/A')}")
                        st.write(f"Recommendation: {feedback_info.get('recommendation', 'N/A')}")
                        st.write(f"Priority: {feedback_info.get('priority', 'N/A')}")
                    else:
                        st.write(f"No specific feedback available for '{label}'.")

                st.subheader("Image with Detections:")
                st.image(image, caption="Image with Detections", use_column_width=True)

            else:
                st.write("No faults detected with the current confidence threshold.")

        except Exception as e:
            st.error(f"Error during inference: {e}")

elif uploaded_file is not None and not session:
    st.warning("ONNX model could not be loaded. Please check the model path and file integrity.")

# Add feedback submission section (optional, requires backend)
# st.subheader("Provide Feedback")
# feedback_text = st.text_area("If you have feedback on the detection results, please enter it here.")
# if st.button("Submit Feedback"):
#     # Here you would add code to save the feedback, e.g., to a file or database
#     st.success("Thank you for your feedback!")
