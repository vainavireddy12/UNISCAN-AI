import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import sqlite3
import re

# ---------------- Flask Config ----------------
app = Flask(__name__)
RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- Load Model ----------------
model = load_model("Models/cnn_fused.h5", compile=False)
CLASS_NAMES = ["Healthy", "Tumor"]
LAST_CONV_LAYER = "conv2d_2"

# ---------------- Image Fusion ----------------
def fuse_ct_mri(ct_path, mri_path):
    ct = cv2.imread(ct_path, 0)
    mri = cv2.imread(mri_path, 0)

    ct = cv2.resize(ct, (128, 128))
    mri = cv2.resize(mri, (128, 128))

    fused = cv2.addWeighted(ct, 0.5, mri, 0.5, 0)
    fused = cv2.cvtColor(fused, cv2.COLOR_GRAY2RGB)

    fused = fused.astype("float32")
    fused = preprocess_input(fused)

    return fused

# ---------------- Grad-CAM ----------------
def make_gradcam(model, img_tensor, class_idx):
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap

# ---------------- Routes ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        ct = request.files["ct"]
        mri = request.files["mri"]

        ct_path = os.path.join(RESULTS_DIR, "ct.png")
        mri_path = os.path.join(RESULTS_DIR, "mri.png")
        ct.save(ct_path)
        mri.save(mri_path)

        # Prediction
        fused_img = fuse_ct_mri(ct_path, mri_path)
        input_tensor = np.expand_dims(fused_img, axis=0)

        pred = model.predict(input_tensor)
        class_idx = np.argmax(pred)
        confidence = float(np.max(pred)) * 100

        # Grad-CAM
        heatmap = make_gradcam(model, input_tensor, class_idx)
        heatmap = cv2.resize(heatmap, (128, 128))
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_JET
        )

        # Undo preprocessing
        fused_vis = fused_img.copy()
        fused_vis -= fused_vis.min()
        fused_vis /= fused_vis.max()
        fused_vis = np.uint8(255 * fused_vis)

        overlay = cv2.addWeighted(fused_vis, 0.6, heatmap_color, 0.4, 0)

        # Save images
        cv2.imwrite(os.path.join(RESULTS_DIR, "fused.png"), fused_vis)
        cv2.imwrite(os.path.join(RESULTS_DIR, "heatmap.png"),
                    cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(RESULTS_DIR, "overlay.png"), overlay)

        return render_template(
            "result.html",
            prediction=CLASS_NAMES[class_idx],
            confidence=round(confidence, 2),
            fused="static/results/fused.png",
            heatmap="static/results/heatmap.png",
            overlay="static/results/overlay.png"
        )

    return render_template("home.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/graphs')
def graphs():
	return render_template('graphs.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')



# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
