import os
import json
import zipfile
import io
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_file, session
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
import docx
from fpdf import FPDF
import tempfile

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Helper: Extract text from uploaded file
def extract_text_from_file(file):
    filename = file.filename.lower()
    if filename.endswith('.txt'):
        return file.read().decode('utf-8')
    elif filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif filename.endswith('.docx'):
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    else:
        raise ValueError("Unsupported file type. Use .txt, .pdf, or .docx")

# Core analysis function using Groq with JSON mode
def analyze_resume(resume_text, job_description=None):
    # Truncate if too long (Groq context limits)
    if len(resume_text) > 10000:
        resume_text = resume_text[:10000] + "\n...[truncated]"

    job_section = ""
    if job_description and job_description.strip():
        job_section = f"Job Role (optional): {job_description}"
    else:
        job_section = "Job Role (optional): Not provided. Evaluate the resume for general suitability."

    system_message = (
        "You are a helpful HR expert. You must output ONLY a valid JSON object. "
        "Do not include any other text, explanations, markdown, or code fences. "
        "The JSON must conform exactly to the requested schema."
    )

    user_prompt = f"""
Analyze the following student resume and provide a structured evaluation.

Resume:
{resume_text}

{job_section}

Return ONLY valid JSON in the following format (no extra text, no markdown):

{{
  "overall_score": number (0-100),
  "strengths": [list of key strengths],
  "weaknesses": [list of issues or missing elements],
  "skills_detected": [list of technical and soft skills],
  "missing_skills": [skills missing based on job role (if job role provided, else empty list)],
  "experience_analysis": "short explanation",
  "project_feedback": "feedback on projects mentioned",
  "resume_tips": [actionable suggestions to improve resume],
  "ats_score": number (0-100),
  "ats_issues": [list of ATS-related problems],
  "final_verdict": "Short hiring recommendation (Reject / Consider / Strong Hire)"
}}

Remember: Output ONLY the JSON object. No introductory or concluding text.
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
            temperature=0.2,
        )
        raw_output = chat_completion.choices[0].message.content.strip()
        
        # Extract JSON: find first '{' and last '}'
        start = raw_output.find('{')
        end = raw_output.rfind('}')
        if start != -1 and end != -1:
            json_str = raw_output[start:end+1]
        else:
            raise ValueError("No JSON object found in response")
        
        result = json.loads(json_str)
        return result
    except Exception as e:
        error_msg = str(e)
        # Provide a user-friendly error
        return {
            "error": f"Analysis failed: {error_msg}",
            "overall_score": 0,
            "strengths": [],
            "weaknesses": ["Could not generate valid analysis. Please try again with a shorter resume or different job description."],
            "skills_detected": [],
            "missing_skills": [],
            "experience_analysis": "N/A",
            "project_feedback": "N/A",
            "resume_tips": ["If error persists, try simplifying the resume text or job description."],
            "ats_score": 0,
            "ats_issues": ["Analysis failed"],
            "final_verdict": "Error"
        }

# Generate PDF report from analysis JSON (unchanged)
def generate_pdf_report(analysis, filename_base):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Resume Analysis Report", ln=1, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Overall Score: {analysis.get('overall_score', 0)}/100", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Final Verdict: {analysis.get('final_verdict', 'N/A')}", ln=1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Strengths:", ln=1)
    pdf.set_font("Arial", size=12)
    for s in analysis.get('strengths', []):
        pdf.cell(200, 6, txt=f"- {s}", ln=1)
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Weaknesses:", ln=1)
    pdf.set_font("Arial", size=12)
    for w in analysis.get('weaknesses', []):
        pdf.cell(200, 6, txt=f"- {w}", ln=1)
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Skills Detected:", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 6, txt=", ".join(analysis.get('skills_detected', [])), ln=1)
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Experience Analysis:", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 6, analysis.get('experience_analysis', 'N/A'))
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Project Feedback:", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 6, analysis.get('project_feedback', 'N/A'))
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Resume Tips:", ln=1)
    pdf.set_font("Arial", size=12)
    for tip in analysis.get('resume_tips', []):
        pdf.cell(200, 6, txt=f"- {tip}", ln=1)
    pdf.ln(3)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="ATS Score & Issues:", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 6, txt=f"ATS Score: {analysis.get('ats_score', 0)}/100", ln=1)
    for issue in analysis.get('ats_issues', []):
        pdf.cell(200, 6, txt=f"- {issue}", ln=1)
    
    # Save to a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp.name)
    temp.close()
    return temp.name

# Routes remain unchanged
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_single():
    data = request.get_json()
    resume_text = data.get('resume_text', '')
    job_description = data.get('job_description', '')
    if not resume_text.strip():
        return jsonify({"error": "Resume text is required"}), 400
    result = analyze_resume(resume_text, job_description)
    if 'history' not in session:
        session['history'] = []
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": "pasted text",
        "verdict": result.get("final_verdict", "N/A"),
        "score": result.get("overall_score", 0),
        "analysis": result
    }
    session['history'].append(history_entry)
    session.modified = True
    return jsonify(result)

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    files = request.files.getlist('files')
    job_description = request.form.get('job_description', '')
    results = []
    for file in files:
        if file.filename == '':
            continue
        try:
            text = extract_text_from_file(file)
            analysis = analyze_resume(text, job_description)
            analysis['filename'] = file.filename
            results.append(analysis)
            if 'history' not in session:
                session['history'] = []
            session['history'].append({
                "timestamp": datetime.now().isoformat(),
                "filename": file.filename,
                "verdict": analysis.get("final_verdict", "N/A"),
                "score": analysis.get("overall_score", 0),
                "analysis": analysis
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    session.modified = True
    return jsonify({"batch_results": results})

@app.route('/download_report_json', methods=['POST'])
def download_report_json():
    data = request.get_json()
    analysis = data.get('analysis')
    filename = data.get('filename', 'resume_analysis')
    if not analysis:
        return jsonify({"error": "No analysis data"}), 400
    json_str = json.dumps(analysis, indent=2)
    return send_file(
        io.BytesIO(json_str.encode('utf-8')),
        mimetype='application/json',
        as_attachment=True,
        download_name=f"{filename}_analysis.json"
    )

@app.route('/download_report_pdf', methods=['POST'])
def download_report_pdf():
    data = request.get_json()
    analysis = data.get('analysis')
    filename = data.get('filename', 'resume_analysis')
    if not analysis:
        return jsonify({"error": "No analysis data"}), 400
    pdf_path = generate_pdf_report(analysis, filename)
    return send_file(pdf_path, as_attachment=True, download_name=f"{filename}_report.pdf")

@app.route('/download_all_reports_zip', methods=['POST'])
def download_all_reports_zip():
    data = request.get_json()
    history = data.get('history', [])
    if not history:
        return jsonify({"error": "No history to export"}), 400
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for idx, entry in enumerate(history):
            analysis = entry.get('analysis')
            if not analysis:
                continue
            fname_base = f"report_{idx+1}_{entry.get('filename', 'unknown')}"
            json_data = json.dumps(analysis, indent=2).encode('utf-8')
            zip_file.writestr(f"{fname_base}.json", json_data)
            pdf_path = generate_pdf_report(analysis, fname_base)
            with open(pdf_path, 'rb') as f:
                zip_file.writestr(f"{fname_base}.pdf", f.read())
            os.unlink(pdf_path)
    zip_buffer.seek(0)
    return send_file(zip_buffer, as_attachment=True, download_name="all_reports.zip")

@app.route('/get_history', methods=['GET'])
def get_history():
    history = session.get('history', [])
    summary = [{"timestamp": h["timestamp"], "filename": h["filename"], "verdict": h["verdict"], "score": h["score"]} for h in history]
    return jsonify({"history": summary})

@app.route('/get_history_full', methods=['GET'])
def get_history_full():
    history = session.get('history', [])
    return jsonify({"full_history": history})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['history'] = []
    session.modified = True
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    app.run(debug=True)