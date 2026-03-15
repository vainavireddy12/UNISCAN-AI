import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, send_file, make_response
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import sqlite3
import re
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from PIL import Image as PILImage

# ---------------- Flask Config ----------------
app = Flask(__name__)
app.secret_key = 'your-secret-key'
RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
DB_NAME = "signup.db"

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

# ---------------- Database Initialization ----------------
def init_predictions_table():
    """Initialize the predictions table if it doesn't exist"""
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    
    # Check if table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    table_exists = cur.fetchone() is not None
    
    if not table_exists:
        # Create table with all columns including patient details
        cur.execute("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                patient_name TEXT,
                age INTEGER,
                gender TEXT,
                patient_id TEXT,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                fused_image_path TEXT NOT NULL,
                heatmap_image_path TEXT NOT NULL,
                overlay_image_path TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """)
        # Create unique index on patient_id
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_patient_id ON predictions(patient_id) WHERE patient_id IS NOT NULL")
    else:
        # Table exists, check and add missing columns
        cur.execute("PRAGMA table_info(predictions)")
        columns = [row[1] for row in cur.fetchall()]
        
        if 'patient_name' not in columns:
            cur.execute("ALTER TABLE predictions ADD COLUMN patient_name TEXT")
        if 'age' not in columns:
            cur.execute("ALTER TABLE predictions ADD COLUMN age INTEGER")
        if 'gender' not in columns:
            cur.execute("ALTER TABLE predictions ADD COLUMN gender TEXT")
        if 'patient_id' not in columns:
            cur.execute("ALTER TABLE predictions ADD COLUMN patient_id TEXT")
            # Create unique index on patient_id (only for non-null values)
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_patient_id ON predictions(patient_id) WHERE patient_id IS NOT NULL")
        else:
            # Column exists, ensure unique index exists
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_patient_id ON predictions(patient_id) WHERE patient_id IS NOT NULL")
    
    con.commit()
    con.close()

# Initialize the table when the app starts
init_predictions_table()

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

        # Get patient details from form
        patient_name = request.form.get("patient_name", "").strip()
        age = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        patient_id = request.form.get("patient_id", "").strip()
        
        # Validate patient ID (now required)
        if not patient_id:
            return render_template("home.html", 
                                 error="Patient ID is required. Please enter a unique Patient ID.")
        
        # Check if patient ID already exists
        con = sqlite3.connect(DB_NAME)
        cur = con.cursor()
        cur.execute("SELECT id FROM predictions WHERE patient_id = ?", (patient_id,))
        existing = cur.fetchone()
        con.close()
        
        if existing:
            # Patient ID already exists, return error
            return render_template("home.html", 
                                 error="Patient ID already exists. Please use a different Patient ID.")
        
        # Get username from session, default to 'guest' if not logged in
        username = session.get('username', 'guest')

        # Create unique filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ct_filename = f"ct_{timestamp}.png"
        mri_filename = f"mri_{timestamp}.png"
        fused_filename = f"fused_{timestamp}.png"
        heatmap_filename = f"heatmap_{timestamp}.png"
        overlay_filename = f"overlay_{timestamp}.png"

        ct_path = os.path.join(RESULTS_DIR, ct_filename)
        mri_path = os.path.join(RESULTS_DIR, mri_filename)
        ct.save(ct_path)
        mri.save(mri_path)

        # Prediction
        fused_img = fuse_ct_mri(ct_path, mri_path)
        input_tensor = np.expand_dims(fused_img, axis=0)

        pred = model.predict(input_tensor)
        class_idx = np.argmax(pred)
        confidence = float(np.max(pred)) * 100
        prediction_result = CLASS_NAMES[class_idx]

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

        # Save images with unique names
        fused_path = os.path.join(RESULTS_DIR, fused_filename)
        heatmap_path = os.path.join(RESULTS_DIR, heatmap_filename)
        overlay_path = os.path.join(RESULTS_DIR, overlay_filename)
        
        cv2.imwrite(fused_path, fused_vis)
        cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
        cv2.imwrite(overlay_path, overlay)

        # Store prediction in database
        fused_image_path = f"static/results/{fused_filename}"
        heatmap_image_path = f"static/results/{heatmap_filename}"
        overlay_image_path = f"static/results/{overlay_filename}"

        # Get local timestamp
        local_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        con = sqlite3.connect(DB_NAME)
        cur = con.cursor()
        cur.execute("""
            INSERT INTO predictions (username, patient_name, age, gender, patient_id, prediction, confidence, fused_image_path, heatmap_image_path, overlay_image_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (username, patient_name if patient_name else None, int(age) if age.isdigit() else None, gender if gender else None, patient_id if patient_id else None, prediction_result, round(confidence, 2), fused_image_path, heatmap_image_path, overlay_image_path, local_timestamp))
        # Get the prediction ID that was just inserted
        prediction_id = cur.lastrowid
        con.commit()
        con.close()

        return render_template(
            "result.html",
            prediction=prediction_result,
            confidence=round(confidence, 2),
            fused=fused_image_path,
            heatmap=heatmap_image_path,
            overlay=overlay_image_path,
            prediction_id=prediction_id,
            patient_name=patient_name,
            age=age,
            gender=gender,
            patient_id=patient_id
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
            session['username'] = 'admin'
            return redirect(url_for('admin_dashboard'))

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            session['username'] = mail1
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/graphs')
def graphs():
	return render_template('graphs.html')

@app.route('/history')
def history():
    # Get username from session
    username = session.get('username', 'guest')
    
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    
    # Fetch predictions for the current user
    cur.execute("""
        SELECT id, prediction, confidence, fused_image_path, heatmap_image_path, overlay_image_path, created_at
        FROM predictions
        WHERE username = ?
        ORDER BY created_at DESC
    """, (username,))
    
    predictions = cur.fetchall()
    con.close()
    
    # Format predictions for template
    history_data = []
    for pred in predictions:
        # Ensure image paths have leading slash for Flask static files
        fused_path = f"/{pred[3]}" if not pred[3].startswith('/') else pred[3]
        heatmap_path = f"/{pred[4]}" if not pred[4].startswith('/') else pred[4]
        overlay_path = f"/{pred[5]}" if not pred[5].startswith('/') else pred[5]
        
        history_data.append({
            'id': pred[0],
            'prediction': pred[1],
            'confidence': pred[2],
            'fused': fused_path,
            'heatmap': heatmap_path,
            'overlay': overlay_path,
            'created_at': pred[6]
        })
    
    return render_template('history.html', predictions=history_data, username=username)

@app.route('/result')
def view_result():
    # Get prediction ID from query parameter
    prediction_id = request.args.get('id')
    
    if not prediction_id:
        return redirect(url_for('history'))
    
    # Get username from session for security check
    username = session.get('username', 'guest')
    is_admin = (username == 'admin')
    
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    
    # Fetch the specific prediction (admin can view any, users can only view their own)
    if is_admin:
        cur.execute("""
            SELECT id, username, patient_name, age, gender, patient_id, prediction, confidence, fused_image_path, heatmap_image_path, overlay_image_path, created_at
            FROM predictions
            WHERE id = ?
        """, (prediction_id,))
    else:
        cur.execute("""
            SELECT id, username, patient_name, age, gender, patient_id, prediction, confidence, fused_image_path, heatmap_image_path, overlay_image_path, created_at
            FROM predictions
            WHERE id = ? AND username = ?
        """, (prediction_id, username))
    
    pred = cur.fetchone()
    con.close()
    
    if not pred:
        if is_admin:
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('history'))
    
    # Ensure image paths have leading slash for Flask static files
    fused_path = f"/{pred[8]}" if not pred[8].startswith('/') else pred[8]
    heatmap_path = f"/{pred[9]}" if not pred[9].startswith('/') else pred[9]
    overlay_path = f"/{pred[10]}" if not pred[10].startswith('/') else pred[10]
    
    # Format prediction data for template
    return render_template(
        "result.html",
        prediction=pred[6],
        confidence=pred[7],
        fused=fused_path,
        heatmap=heatmap_path,
        overlay=overlay_path,
        prediction_id=prediction_id,
        patient_name=pred[2] or '',
        age=pred[3] or '',
        gender=pred[4] or '',
        patient_id=pred[5] or ''
    )

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route('/admin')
def admin_dashboard():
    # Check if user is admin
    if session.get('username') != 'admin':
        return redirect(url_for('index'))
    
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    
    # Fetch all users
    cur.execute("SELECT user, name, email, mobile FROM info ORDER BY user")
    users = cur.fetchall()
    
    # Fetch all predictions for statistics
    cur.execute("""
        SELECT prediction
        FROM predictions
    """)
    all_predictions = cur.fetchall()
    con.close()
    
    # Format users data
    users_data = []
    for user in users:
        users_data.append({
            'username': user[0],
            'name': user[1],
            'email': user[2],
            'mobile': user[3]
        })
    
    # Count statistics
    total_users = len(users_data)
    total_predictions = len(all_predictions)
    healthy_count = sum(1 for p in all_predictions if p[0] == 'Healthy')
    tumor_count = sum(1 for p in all_predictions if p[0] == 'Tumor')
    
    return render_template('admin.html', 
                         users=users_data,
                         total_users=total_users,
                         total_predictions=total_predictions,
                         healthy_count=healthy_count,
                         tumor_count=tumor_count)

@app.route('/admin/user/<username>')
def admin_user_history(username):
    # Check if user is admin
    if session.get('username') != 'admin':
        return redirect(url_for('index'))
    
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    
    # Get user details
    cur.execute("SELECT user, name, email, mobile FROM info WHERE user = ?", (username,))
    user_data = cur.fetchone()
    
    if not user_data:
        con.close()
        return redirect(url_for('admin_dashboard'))
    
    # Get user's predictions
    cur.execute("""
        SELECT id, prediction, confidence, fused_image_path, heatmap_image_path, overlay_image_path, created_at
        FROM predictions
        WHERE username = ?
        ORDER BY created_at DESC
    """, (username,))
    
    predictions = cur.fetchall()
    con.close()
    
    # Format user data
    user_info = {
        'username': user_data[0],
        'name': user_data[1],
        'email': user_data[2],
        'mobile': user_data[3]
    }
    
    # Format predictions data
    predictions_data = []
    for pred in predictions:
        # Ensure image paths have leading slash for Flask static files
        fused_path = f"/{pred[3]}" if not pred[3].startswith('/') else pred[3]
        heatmap_path = f"/{pred[4]}" if not pred[4].startswith('/') else pred[4]
        overlay_path = f"/{pred[5]}" if not pred[5].startswith('/') else pred[5]
        
        predictions_data.append({
            'id': pred[0],
            'prediction': pred[1],
            'confidence': pred[2],
            'fused': fused_path,
            'heatmap': heatmap_path,
            'overlay': overlay_path,
            'created_at': pred[6]
        })
    
    return render_template('admin_user_history.html', 
                         user=user_info, 
                         predictions=predictions_data)

@app.route('/generate-report')
def generate_report():
    """Generate PDF report for a prediction"""
    prediction_id = request.args.get('id')
    
    if not prediction_id:
        return redirect(url_for('history'))
    
    # Get username from session for security check
    username = session.get('username', 'guest')
    is_admin = (username == 'admin')
    
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()
    
    # Fetch the specific prediction
    if is_admin:
        cur.execute("""
            SELECT id, username, patient_name, age, gender, patient_id, prediction, confidence, fused_image_path, heatmap_image_path, overlay_image_path, created_at
            FROM predictions
            WHERE id = ?
        """, (prediction_id,))
    else:
        cur.execute("""
            SELECT id, username, patient_name, age, gender, patient_id, prediction, confidence, fused_image_path, heatmap_image_path, overlay_image_path, created_at
            FROM predictions
            WHERE id = ? AND username = ?
        """, (prediction_id, username))
    
    pred = cur.fetchone()
    con.close()
    
    if not pred:
        return redirect(url_for('history'))
    
    # Get image paths
    fused_path = pred[8] if pred[8].startswith('/') else f"/{pred[8]}"
    heatmap_path = pred[9] if pred[9].startswith('/') else f"/{pred[9]}"
    overlay_path = pred[10] if pred[10].startswith('/') else f"/{pred[10]}"
    
    # Create PDF with patient details
    buffer = create_pdf_report(
        pred[2], pred[3], pred[4], pred[5],  # patient_name, age, gender, patient_id
        pred[6], pred[7],  # prediction, confidence
        fused_path, heatmap_path, overlay_path, 
        pred[11], pred[0]  # created_at, report_id
    )
    
    # Create response
    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=UNISCAN_Report_{pred[0]}.pdf'
    
    buffer.close()
    return response

def create_pdf_report(patient_name, age, gender, patient_id, prediction, confidence, fused_path, heatmap_path, overlay_path, created_at, report_id):
    """Create PDF report with all details and images"""
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#0d6efd'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#212529'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#495057'),
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    # Build PDF content
    story = []
    
    # Header
    story.append(Paragraph("UNISCAN AI", title_style))
    story.append(Paragraph("Unified Diagnostic Report Generator", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Details Section at the top
    story.append(Paragraph("Patient Information", heading_style))
    patient_data = []
    if patient_name:
        patient_data.append(['Patient Name', patient_name])
    if age:
        patient_data.append(['Age', str(age)])
    if gender:
        patient_data.append(['Gender', gender])
    if patient_id:
        patient_data.append(['Patient ID', patient_id])
    patient_data.append(['Report ID', f'#{report_id}'])
    patient_data.append(['Generated on', created_at])
    
    if patient_data:
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Diagnostic Result
    story.append(Paragraph("Diagnostic Result", heading_style))
    result_color = colors.HexColor('#198754') if prediction == 'Healthy' else colors.HexColor('#dc3545')
    result_style = ParagraphStyle('ResultStyle', parent=styles['Heading2'], fontSize=18, textColor=result_color, alignment=TA_CENTER)
    story.append(Paragraph(f"<b>{prediction}</b>", result_style))
    story.append(Paragraph(f"Confidence: <b>{confidence}%</b>", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Image Analysis Section
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Image Analysis", heading_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Image Subheading Style
    image_subheading_style = ParagraphStyle(
        'ImageSubheading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#0d6efd'),
        spaceAfter=8,
        spaceBefore=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Image Description Style
    image_desc_style = ParagraphStyle(
        'ImageDesc',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#6c757d'),
        alignment=TA_CENTER,
        spaceAfter=10
    )
    
    # Fused CT-MRI Image
    try:
        img_path = fused_path.lstrip('/')
        if not os.path.exists(img_path):
            img_path = fused_path if not fused_path.startswith('/') else fused_path[1:]
        if os.path.exists(img_path):
            img = Image(img_path, width=4.5*inch, height=4.5*inch, kind='proportional')
            # Center the image
            img_table = Table([[img]], colWidths=[6*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 0.15*inch))
            story.append(Paragraph("1. Fused CT-MRI Image", image_subheading_style))
            story.append(Paragraph("Combined CT and MRI scan fusion for enhanced diagnostic clarity", image_desc_style))
        else:
            story.append(Paragraph("Fused CT-MRI Image: File not found", styles['Normal']))
    except Exception as e:
        story.append(Paragraph(f"Fused Image not available: {str(e)}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Grad-CAM Heatmap
    try:
        img_path = heatmap_path.lstrip('/')
        if not os.path.exists(img_path):
            img_path = heatmap_path if not heatmap_path.startswith('/') else heatmap_path[1:]
        if os.path.exists(img_path):
            img = Image(img_path, width=4.5*inch, height=4.5*inch, kind='proportional')
            img_table = Table([[img]], colWidths=[6*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 0.15*inch))
            story.append(Paragraph("2. Grad-CAM Heatmap", image_subheading_style))
            story.append(Paragraph("Visualization of regions of interest that influence the AI model's prediction", image_desc_style))
        else:
            story.append(Paragraph("Grad-CAM Heatmap: File not found", styles['Normal']))
    except Exception as e:
        story.append(Paragraph(f"Heatmap not available: {str(e)}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Overlay Analysis
    try:
        img_path = overlay_path.lstrip('/')
        if not os.path.exists(img_path):
            img_path = overlay_path if not overlay_path.startswith('/') else overlay_path[1:]
        if os.path.exists(img_path):
            img = Image(img_path, width=4.5*inch, height=4.5*inch, kind='proportional')
            img_table = Table([[img]], colWidths=[6*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 0.15*inch))
            story.append(Paragraph("3. Overlay Analysis", image_subheading_style))
            story.append(Paragraph("Combined visualization of fused image with heatmap overlay for comprehensive analysis", image_desc_style))
        else:
            story.append(Paragraph("Overlay Analysis: File not found", styles['Normal']))
    except Exception as e:
        story.append(Paragraph(f"Overlay not available: {str(e)}", styles['Normal']))
    
    story.append(Spacer(1, 0.4*inch))
    story.append(PageBreak())
    
    # Report Summary
    story.append(Paragraph("Report Summary", heading_style))
    summary_data = [
        ['Prediction', prediction],
        ['Confidence Level', f'{confidence}%'],
        ['Image Modalities', 'CT Scan + MRI'],
        ['Analysis Status', 'Completed'],
        ['Report Date', created_at]
    ]
    summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations and Precautions
    story.append(Paragraph("Recommendations & Precautions", heading_style))
    
    if prediction == 'Healthy':
        story.append(Paragraph("<b>Recommendations:</b>", styles['Normal']))
        recommendations = [
            "Continue regular health monitoring and follow-up appointments",
            "Maintain a healthy lifestyle with balanced diet and regular exercise",
            "Schedule routine medical check-ups as recommended by your physician",
            "Stay hydrated and get adequate rest",
            "Keep a record of your medical history for future reference"
        ]
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", normal_style))
        
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("<b>Precautions:</b>", styles['Normal']))
        precautions = [
            "Monitor for any new or unusual symptoms",
            "Consult a healthcare professional if you experience any changes",
            "Follow your doctor's advice regarding follow-up imaging if needed",
            "Maintain regular communication with your healthcare provider",
            "This AI analysis is a screening tool and should not replace professional medical diagnosis"
        ]
        for prec in precautions:
            story.append(Paragraph(f"• {prec}", normal_style))
    else:
        story.append(Paragraph("<b>Immediate Recommendations:</b>", styles['Normal']))
        recommendations = [
            "Schedule an immediate consultation with a neurologist or oncologist",
            "Bring this report and all medical imaging to your appointment",
            "Discuss treatment options and next steps with your healthcare team",
            "Consider seeking a second opinion from a specialist",
            "Maintain all medical records and imaging studies"
        ]
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", normal_style))
        
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("<b>Important Precautions:</b>", styles['Normal']))
        precautions = [
            "Do not delay seeking professional medical attention",
            "This AI analysis is a screening tool and requires clinical confirmation",
            "Avoid self-diagnosis or self-treatment",
            "Follow your doctor's recommendations for additional tests or procedures",
            "Keep family members informed and seek emotional support if needed"
        ]
        for prec in precautions:
            story.append(Paragraph(f"• {prec}", normal_style))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<i>This report is generated by UNISCAN AI and should be reviewed by a qualified medical professional.</i>", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
