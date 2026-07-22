from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import os
import cv2

from config import UPLOAD_DIR, CTR_MODEL, BI_MODEL
from predictor import DetectronPredictor
from services.ctr_service import CTRService
from services.bi_service import BIService
from utils.file_utils import save_file
from utils.image_utils import read_image

app = Flask(__name__)

# -------------------------------------------------------
# Ensure upload directory exists
# -------------------------------------------------------

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------------
# Load models once (important for performance)
# -------------------------------------------------------

ctr_predictor = DetectronPredictor(
    config_path=CTR_MODEL["config_path"],
    weights_path=str(CTR_MODEL["weights_path"]),
)

bi_predictor = DetectronPredictor(
    config_path=BI_MODEL["config_path"],
    weights_path=str(BI_MODEL["weights_path"]),
)

ctr_service = CTRService(ctr_predictor)
bi_service = BIService(bi_predictor)

# -------------------------------------------------------
# Home
# -------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------------------------------
# CTR
# -------------------------------------------------------

@app.route("/ctr", methods=["GET", "POST"])
def ctr():

    if request.method == "GET":
        return render_template("ctr.html")

    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    if file.filename == "":
        return "Empty filename", 400

    try:

        # Save uploaded image
        input_path = save_file(file, UPLOAD_DIR)

        image = read_image(input_path)

        # Run inference
        result = ctr_service.predict(image)

        # Save annotated image
        output_filename = "output_" + secure_filename(file.filename)

        output_path = os.path.join(
            UPLOAD_DIR,
            output_filename,
        )

        cv2.imwrite(
            output_path,
            result["image"],
        )

        return render_template(
            "ctr.html",
            prediction=result,
            img_path=url_for(
                "static",
                filename=f"images/{output_filename}",
            ),
        )

    except Exception as e:
        return {"error": str(e)}, 500


# -------------------------------------------------------
# BI
# -------------------------------------------------------

@app.route("/bi", methods=["GET", "POST"])
def bi():

    if request.method == "GET":
        return render_template("bi.html")

    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    if file.filename == "":
        return "Empty filename", 400

    try:

        # Save uploaded image
        input_path = save_file(file, UPLOAD_DIR)

        image = read_image(input_path)

        # Run inference
        result = bi_service.predict(image)

        # Save annotated image
        output_filename = "output_" + secure_filename(file.filename)

        output_path = os.path.join(
            UPLOAD_DIR,
            output_filename,
        )

        cv2.imwrite(
            output_path,
            result["image"],
        )

        return render_template(
            "bi.html",
            prediction=result,
            img_path=url_for(
                "static",
                filename=f"images/{output_filename}",
            ),
        )

    except Exception as e:
        return {"error": str(e)}, 500


# -------------------------------------------------------
# Run application
# -------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )