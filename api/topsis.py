from flask import Flask, request, jsonify
import io
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
import numpy as np

app = Flask(__name__)


def run_topsis(csv_content, weights_str, impacts_str):
    """
    Run TOPSIS algorithm on the provided CSV content.
    Returns the result as CSV string.
    """
    # Read CSV from string
    data = pd.read_csv(io.StringIO(csv_content))
    
    if data.shape[1] < 3:
        raise Exception("Input file must contain at least 3 columns")
    
    criteria_data = data.iloc[:, 1:]
    
    # Parse weights and impacts
    weights = np.array([w.strip() for w in weights_str.split(",")], dtype=float)
    impacts = [i.strip() for i in impacts_str.split(",")]
    
    if len(weights) != criteria_data.shape[1]:
        raise Exception(f"Weights count ({len(weights)}) must match criteria columns ({criteria_data.shape[1]})")
    
    if len(impacts) != criteria_data.shape[1]:
        raise Exception(f"Impacts count ({len(impacts)}) must match criteria columns ({criteria_data.shape[1]})")
    
    for imp in impacts:
        if imp not in ["+", "-"]:
            raise Exception("Impacts must be + or -")
    
    # Normalize matrix
    norm_matrix = criteria_data / np.sqrt((criteria_data ** 2).sum())
    
    # Weighted matrix
    weighted_matrix = norm_matrix * weights
    
    # Ideal best and worst
    ideal_best, ideal_worst = [], []
    
    for i in range(criteria_data.shape[1]):
        if impacts[i] == "+":
            ideal_best.append(weighted_matrix.iloc[:, i].max())
            ideal_worst.append(weighted_matrix.iloc[:, i].min())
        else:
            ideal_best.append(weighted_matrix.iloc[:, i].min())
            ideal_worst.append(weighted_matrix.iloc[:, i].max())
    
    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)
    
    # Calculate distances
    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    
    # Calculate score and rank
    score = dist_worst / (dist_best + dist_worst)
    rank = score.rank(ascending=False)
    
    data["Topsis Score"] = score
    data["Rank"] = rank.astype(int)
    
    # Return as CSV string
    return data.to_csv(index=False)


def send_email(to_email, result_csv, original_filename):
    """
    Send the result CSV via email.
    """
    smtp_host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    smtp_user = os.environ.get('SMTP_USER', '')
    smtp_pass = os.environ.get('SMTP_PASS', '')
    
    if not smtp_user or not smtp_pass:
        raise Exception("Email configuration not set. Please configure SMTP_USER and SMTP_PASS environment variables.")
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg['Subject'] = 'TOPSIS Analysis Results'
    
    # Email body
    body = """
Hello,

Your TOPSIS analysis has been completed successfully.

Please find the results attached as a CSV file. The file includes:
- Original data columns
- Topsis Score (higher is better)
- Rank (1 = best alternative)

Thank you for using our TOPSIS Web Service!

Best regards,
TOPSIS Web Service
"""
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach CSV file
    result_filename = original_filename.replace('.csv', '_result.csv') if original_filename else 'topsis_result.csv'
    attachment = MIMEBase('text', 'csv')
    attachment.set_payload(result_csv)
    encoders.encode_base64(attachment)
    attachment.add_header('Content-Disposition', f'attachment; filename="{result_filename}"')
    msg.attach(attachment)
    
    # Send email
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


@app.route('/api/topsis', methods=['POST'])
def topsis_handler():
    try:
        # Get file
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read file content
        file_content = file.read().decode('utf-8')
        file_name = file.filename
        
        # Get form fields
        weights = request.form.get('weights', '')
        impacts = request.form.get('impacts', '')
        email = request.form.get('email', '')
        
        # Validate inputs
        if not weights:
            return jsonify({'success': False, 'error': 'Weights are required'}), 400
        if not impacts:
            return jsonify({'success': False, 'error': 'Impacts are required'}), 400
        if not email:
            return jsonify({'success': False, 'error': 'Email is required'}), 400
        
        # Run TOPSIS
        result_csv = run_topsis(file_content, weights, impacts)
        
        # Send email
        send_email(email, result_csv, file_name)
        
        return jsonify({
            'success': True,
            'message': f'TOPSIS analysis completed! Results sent to {email}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/topsis', methods=['GET'])
def topsis_info():
    return jsonify({'message': 'TOPSIS API is running. Use POST to submit analysis.'})


# For Vercel
app = app
