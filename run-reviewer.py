import os
import re
from datetime import datetime

from PyPDF2 import PdfReader
import tiktoken
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# === CONFIG ===
client = OpenAI(api_key="your openai-key")  # 🔐 Replace this securely in production
MODEL = "gpt-4"
MAX_TOKENS = 3000  # Leave room for long answers

# === Load PDF and extract text ===
def load_pdf(path):
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

# === Crude section splitter based on headings ===
def split_by_sections(text):
    sections = re.split(r"\n(?=\d?\s?[A-Z][^\n]{1,80})", text)  # crude heading-based split
    return [s.strip() for s in sections if len(s.strip()) > 100]

# === Token length check ===
def num_tokens_from_string(string: str, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))

# === GPT-4 reviewer function ===
def review_section(section_text, section_title):
    prompt = f"""
You are an expert reviewer for top-tier AI conferences (e.g., NeurIPS, ICLR, ICML).
Evaluate the following paper section titled "{section_title}" in terms of:

1. Clarity of writing
2. Novelty and originality
3. Technical depth and correctness
4. Suggestions for improvement

Return a structured review in bullet points.

SECTION:
\"\"\"
{section_text}
\"\"\"
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful and critical AI research paper reviewer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.3,
    )
    return response.choices[0].message.content

# === Save all reviews to a PDF ===
def save_reviews_to_pdf(review_dict, output_path="gpt4_review_feedback.pdf"):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    x_margin = 50
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x_margin, y, " GPT-4 Research Paper Review")
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(x_margin, y, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 40

    for section, review in review_dict.items():
        if y < 100:
            c.showPage()
            y = height - 50

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x_margin, y, f" Section: {section}")
        y -= 20

        c.setFont("Helvetica", 10)
        for line in review.split("\n"):
            if y < 100:
                c.showPage()
                y = height - 50
            c.drawString(x_margin, y, line.strip()[:110])  # Clip long lines
            y -= 15
        y -= 20

    c.save()
    print(f"\n PDF review saved as: {output_path}")

# === Main controller ===
def run_reviewer(paper_path):
    text = load_pdf(paper_path)
    sections = split_by_sections(text)
    review_dict = {}

    for section in sections:
        tokens = num_tokens_from_string(section)
        if tokens < 3500:
            print("\n" + "="*80)
            title = section.split('\n')[0].strip()
            print(f" Reviewing Section: {title}")
            review = review_section(section, title)
            review_dict[title] = review
            print(" GPT-4 Review:\n", review)
        else:
            print(f" Skipped long section ({tokens} tokens): {section[:60]}...")

    # Save all reviews to a PDF
    save_reviews_to_pdf(review_dict)

# === Run Script ===
if __name__ == "__main__":
    run_reviewer("MedicalGAT_Paper_arXiv.pdf")  # Replace with the path to your paper
