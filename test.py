import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('final_brain_tumor_model_vgg16V2.h5')

labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

tumor_images = {
    "glioma_tumor": "static/images/glioma_tumor.jpg",
    "meningioma_tumor": "static/images/meningioma_tumor.jpg",
    "no_tumor": "static/images/no_tumor.jpg",
    "pituitary_tumor": "static/images/pituitary_tumor.jpg"
}

tumor_context = {
    "glioma_tumor": "Glioma tumors are a type of brain tumor that arises from glial cells, which provide support and protection for neurons. They are classified into several grades, with high-grade gliomas being more aggressive and difficult to treat. Gliomas can occur in various regions of the brain and spinal cord, and symptoms may include headaches, seizures, and neurological deficits, depending on the tumor's location. Treatment options often involve surgery, radiation therapy, and chemotherapy, though outcomes vary based on tumor grade and location.",
    "meningioma_tumor": "Meningioma tumors are non-cancerous growths that develop in the meninges, the protective layers surrounding the brain and spinal cord. These tumors are generally slow-growing and may cause symptoms like headaches, vision problems, or seizures, depending on their size and location. Though most meningiomas are benign, some may be malignant and require aggressive treatment, including surgery, radiation, or chemotherapy. Regular monitoring through imaging is crucial to track the tumor's growth and determine the appropriate treatment approach.",
    "no_tumor": "No tumor detected in the MRI scan. However, consult a medical professional for a comprehensive diagnosis.",
    "pituitary_tumor": "Pituitary tumors are abnormal growths in the pituitary gland, which is located at the base of the brain and regulates hormone production. These tumors can be **benign** (non-cancerous) or **malignant** (cancerous). Symptoms may vary depending on the tumor's size and type, including hormonal imbalances, vision problems, and headaches. The treatment typically involves surgery, radiation, or medications to control hormone production. Despite being mostly non-cancerous, pituitary tumors can still lead to serious complications if left untreated."
}

image_size = (150, 150)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, image_size)
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  

        prediction = model.predict(img_array)
        print(prediction)
        predicted_class_index = prediction.argmax()  
        predicted_class_label = labels[predicted_class_index]

        tumor_image = tumor_images.get(predicted_class_label, "static/images")

        tumor_description = tumor_context.get(predicted_class_label, "Description not available.")

        return jsonify({
            "prediction": predicted_class_label,
            "confidence": float(prediction[0][predicted_class_index]),
            "tumor_image": tumor_image,
            "description": tumor_description
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)