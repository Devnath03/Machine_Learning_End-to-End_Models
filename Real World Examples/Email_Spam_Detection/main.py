# Import required libraries
import os
from flask import Flask, request, render_template, redirect, url_for
import pickle

# Extra libraries for reading documents
import docx  # For .docx files
import PyPDF2  # For PDF files

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the pre-trained spam detection model pipeline
pipe = pickle.load(open('model.pickle', 'rb'))

def read_file_content(file_path):
    """
    Reads text content from different file formats.
    Supports .txt, .pdf, and .docx files.
    """
    text = ""

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    elif file_path.endswith(".pdf"):
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"Error reading PDF: {e}")

    elif file_path.endswith(".docx"):
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX: {e}")

    return text.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Home page route.
    - Displays form for direct email text input or file upload.
    """
    email_text = ""
    is_spam = None
    file_result = None

    if request.method == 'POST':
        # If direct text input is provided
        email_text = request.form.get('email_text', '').strip()
        if email_text:
            prediction = pipe.predict([email_text])[0]
            is_spam = bool(prediction)

    return render_template('index.html', email_text=email_text, is_spam=is_spam, file_result=file_result)

@app.route('/scan-file', methods=['POST'])
def scan_file():
    """
    Scans uploaded file to check if its content is spam.
    """
    file = request.files.get('file')
    file_result = None

    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Read file content
        file_content = read_file_content(file_path)

        if file_content:
            prediction = pipe.predict([file_content])[0]
            file_result = bool(prediction)  # True if spam

    return render_template('index.html', email_text="", is_spam=None, file_result=file_result)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
