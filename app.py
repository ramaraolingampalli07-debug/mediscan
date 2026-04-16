from flask import Flask, request, jsonify, render_template, send_from_directory
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import os
import re
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# Load medicine dataset
df = pd.read_csv('medicines.csv')
df_lower = df.copy()
for col in df_lower.select_dtypes(include='object').columns:
    df_lower[col] = df_lower[col].str.lower().fillna('')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Enhance image for better OCR accuracy"""
    # Convert to grayscale
    image = image.convert('L')
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    # Sharpen
    image = image.filter(ImageFilter.SHARPEN)
    # Resize if too small
    w, h = image.size
    if w < 800:
        scale = 800 / w
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image

def extract_text(image_path):
    """Extract text using Tesseract with multiple configs"""
    img = Image.open(image_path)
    img_processed = preprocess_image(img)
    
    texts = []
    configs = [
        r'--oem 3 --psm 6',
        r'--oem 3 --psm 11',
        r'--oem 3 --psm 3',
    ]
    
    for cfg in configs:
        try:
            text = pytesseract.image_to_string(img_processed, config=cfg)
            texts.append(text)
        except:
            pass
    
    # Combine unique lines
    all_lines = set()
    for t in texts:
        for line in t.split('\n'):
            stripped = line.strip()
            if len(stripped) > 2:
                all_lines.add(stripped)
    
    combined = '\n'.join(sorted(all_lines, key=len, reverse=True))
    return combined, img_processed

def clean_text(text):
    """Clean and normalize OCR output"""
    # Remove special chars except common medicine-related ones
    text = re.sub(r'[^\w\s\.\+\-\%\/]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def extract_keywords(text):
    """Extract potential medicine names and compositions"""
    cleaned = clean_text(text)
    words = cleaned.split()
    
    # Filter meaningful words (>2 chars, not pure numbers)
    keywords = []
    for word in words:
        if len(word) > 2 and not word.isdigit():
            keywords.append(word)
    
    # Also extract multi-word phrases (2-3 words)
    tokens = cleaned.split()
    phrases = []
    for i in range(len(tokens) - 1):
        if len(tokens[i]) > 2 and len(tokens[i+1]) > 2:
            phrases.append(f"{tokens[i]} {tokens[i+1]}")
    for i in range(len(tokens) - 2):
        if len(tokens[i]) > 2:
            phrases.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
    
    return list(set(keywords + phrases))

def match_medicines(extracted_text):
    """Match extracted text against medicine database"""
    keywords = extract_keywords(extracted_text)
    matches = []
    seen_indices = set()
    
    search_columns = ['Medicine Name', 'Generic Name', 'Composition', 'Drug Class']
    
    for keyword in keywords:
        if len(keyword) < 3:
            continue
        
        for _, row in df.iterrows():
            idx = row.name
            if idx in seen_indices:
                continue
            
            score = 0
            matched_fields = []
            
            for col in search_columns:
                cell_val = str(row[col]).lower()
                if keyword in cell_val:
                    weight = {'Medicine Name': 4, 'Generic Name': 3, 'Composition': 3, 'Drug Class': 1}.get(col, 1)
                    score += weight
                    matched_fields.append(col)
            
            if score > 0:
                matches.append({
                    'index': idx,
                    'score': score,
                    'matched_fields': matched_fields,
                    'data': row.to_dict()
                })
                seen_indices.add(idx)
    
    # Sort by score descending
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    # Deduplicate and return top results
    results = []
    for m in matches[:10]:
        results.append({
            'score': m['score'],
            'matched_on': list(set(m['matched_fields'])),
            'medicine_name': m['data']['Medicine Name'],
            'generic_name': m['data']['Generic Name'],
            'composition': m['data']['Composition'],
            'drug_class': m['data']['Drug Class'],
            'manufacturer': m['data']['Manufacturer'],
            'dosage_form': m['data']['Dosage Form'],
            'strength': m['data']['Strength'],
            'uses': m['data']['Uses'],
            'side_effects': m['data']['Side Effects'],
            'prescription': m['data']['Prescription Required'],
            'age_group': m['data'].get('Age Group', 'Not specified'),
            'min_age': m['data'].get('Min Age', 'N/A'),
            'age_notes': m['data'].get('Age Notes', '')
        })
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, BMP, or TIFF'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Extract text
        raw_text, _ = extract_text(filepath)
        
        if not raw_text.strip():
            return jsonify({
                'raw_text': '',
                'keywords': [],
                'matches': [],
                'warning': 'No text could be extracted from the image. Try a clearer image.'
            })
        
        keywords = extract_keywords(raw_text)
        matches = match_medicines(raw_text)
        
        return jsonify({
            'raw_text': raw_text,
            'keywords': keywords[:20],
            'matches': matches,
            'total_matches': len(matches)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/dataset-stats')
def dataset_stats():
    return jsonify({
        'total_medicines': len(df),
        'drug_classes': df['Drug Class'].nunique(),
        'manufacturers': df['Manufacturer'].nunique(),
        'dosage_forms': df['Dosage Form'].unique().tolist()
    })

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)