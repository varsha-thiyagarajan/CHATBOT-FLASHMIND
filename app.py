import os, io, base64, re, json, logging
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from PIL import Image
import pytesseract
from groq import Groq
from pdf2image import convert_from_bytes
import PyPDF2
import requests

# ----------------------------
# Basic config & logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flashmind")

# ----------------------------
# Tesseract config (Windows optional)
# ----------------------------
TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.name == "nt" and os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ----------------------------
# Groq API
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "********************************************")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set. Set it in your environment.")
client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change_me_in_env")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024   # 25MB upload guard
CORS(app, resources={r"/*": {"origins": "*"}})

# ----------------------------
# LLM helpers
# ----------------------------
def groq_task(system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 4000) -> str:
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content.strip() if completion.choices else "⚠️ No response generated."
    except Exception as e:
        logger.exception("groq_task error")
        return f"⚠️ Chat error: {e}"

def groq_chat(prompt: str, temperature: float = 0.4, max_tokens: int = 2000) -> str:
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a concise, factual assistant. "
                    "Do not use conversational greetings or pleasantries. "
                    "Answer user questions directly and without introduction. "
                    "If the input is informational, provide a crisp summary (5-7 key bullet points). "
                    "Ensure responses are straight to the point."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content.strip() if completion.choices else "⚠️ No response generated."
    except Exception as e:
        logger.exception("groq_chat error")
        return f"⚠️ Chat error: {e}"

# ----------------------------
# Intent classification
# ----------------------------
QUESTION_HINTS = ("?", "q1", "q2", "question", "questions", "explain", "why", "how")

def classify_intent(text: str) -> str:
    lowered = text.lower()
    if any(h in lowered for h in QUESTION_HINTS):
        return "Question"
    return "Informational"

def detect_options(user_text: str) -> dict:
    t = user_text.lower()
    return {
        "want_explain": any(k in t for k in ["explain", "step by step", "step-by-step", "steps", "detailed"]),
        "want_notes": any(k in t for k in ["notes", "short notes", "exam", "key points", "revision points"])
    }

# ----------------------------
# Problem-solving helpers
# ----------------------------
def generate_hints_and_solution(problem_text: str):
    system_prompt = (
        "You are a helpful tutor. A student has a math or logical problem to solve. "
        "Generate a complete, step-by-step solution. "
        "Also, create exactly 6 hints to guide the student, each progressively more helpful. "
        "The hints should follow this strict order:\n"
        "1. **Rephrasing:** Rephrase the problem in simpler terms to clarify what is being asked.\n"
        "2. **Formula/Concept:** Remind the student of the relevant formula, theorem, or concept without giving the solution.\n"
        "3. **First Step:** Guide the student through the very first step of the solution process.\n"
        "4. **Common Mistake:** Point out a common mistake or pitfall students might encounter when solving this type of problem.\n"
        "5. **Similar Problem:** Link the problem to a similar type the student might have seen before to encourage pattern recognition.\n"
        "6. **Analytical Question:** Ask a guiding question that prompts the student to think analytically about the next step.\n"
        "Format the output strictly as a JSON object with 'solution' and 'hints' keys. "
        "The 'hints' value should be a list of exactly 6 strings, one for each hint type."
    )
    user_prompt = f"Generate hints and a solution for the following problem:\n\n{problem_text}"
    response = groq_task(system_prompt, user_prompt, temperature=0.5, max_tokens=1500)

    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group(0))
        return None
    except json.JSONDecodeError:
        logger.warning("Failed to decode JSON from AI response (math/logic). Raw: %s", response[:300])
        return None

def generate_coding_problem_response(problem_text: str):
    system_prompt = (
        "You are an expert programming tutor. A student has a coding problem. "
        "Generate a clear and concise solution in Python, including the code and an explanation. "
        "Also, create 6 hints, ranging from conceptual to specific implementation details. "
        "The hints should guide the user without giving away the final code. "
        "Format the output strictly as a JSON object with 'solution' and 'hints' keys. "
        "The 'solution' key should contain the code block and a brief explanation. "
        "The 'hints' value should be a list of 6 strings."
    )
    user_prompt = f"Generate hints and a Python solution for the following coding problem:\n\n{problem_text}"
    response = groq_task(system_prompt, user_prompt, temperature=0.5, max_tokens=2000)

    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group(0))
        return None
    except json.JSONDecodeError:
        logger.warning("Failed to decode JSON from AI response (coding). Raw: %s", response[:300])
        return None

# ----------------------------
# QA processor
# ----------------------------
def process_qa_from_text(source_text: str, want_explain: bool, want_notes: bool) -> str:
    system_prompt = (
        "You are an expert tutor. Extract questions from the source and answer each directly. "
        "Only provide questions that actually exist in the source text. "
        "If no questions exist in the source and the user did not ask any question, "
        "produce a 5–7 bullet summary of the source instead."
    )
    user_prompt = (
        f"<Source>\n{source_text}\n</Source>\n\n"
        "Task:\n"
        "- Extract only questions present in <Source>.\n"
        "- Provide direct, concise answers in 1–3 sentences.\n"
    )
    if want_explain:
        user_prompt += "- Provide step-by-step explanations if requested.\n"
    if want_notes:
        user_prompt += "- Provide short notes if requested.\n"
    user_prompt += (
        "\nIf there are no questions in <Source>, produce a 5–7 bullet summary.\n"
        "Output clearly in markdown."
    )
    return groq_task(system_prompt, user_prompt, temperature=0.2, max_tokens=8000)

# ----------------------------
# Routes
# ----------------------------
@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/")
def home():
    session.pop("last_file_text", None)
    session.pop("problem_hints", None)
    session.pop("problem_solution", None)
    session.pop("hint_count", None)
    session.pop("last_problem", None)
    try:
        return render_template("flashmind_index.html")
    except Exception:
        return "FlashMind Server Running", 200

@app.route("/explain_more", methods=["POST"])
def explain_more():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    text = session.get("last_file_text", "")

    if not text and not user_message:
        return jsonify({"reply": "⚠️ No context. Upload a file or type text before requesting a detailed explanation.", "detail": False})

    combined = (f"{user_message}\n\n{text}").strip() if text else user_message

    system_prompt = (
        "You are an expert teacher. Using ONLY the provided content as context, produce a clear, student-friendly explanation. "
        "Format rules: "
        "- Use plain text only (no markdown, no bullets, no numbering inside headings). "
        "- Always include the exact 7 section headings in this order: "
        "Topic name, Usage, Core Idea, Clarity, Step by Step example, Practical Use, Check & Practice. "
        "- After each heading, leave a blank line and then provide the content. "
        "- For Topic Name, provide the title of the content. "
        "- For Usage, explain the real-world usage and need for it, also give one real example. "
        "- For Core Idea (What), give a short, simple definition or analogy (2–4 sentences). "
        "- For Clarity (How), explain step by step (up to 5 steps) and include a tiny pseudocode or text diagram if useful. "
        "- For Step by Step Example, provide worked-out examples that gradually increase in difficulty. "
        "- For Practical Use, explain where it is applied in real life or jobs. "
        "- For Check & Practice (Do), write exactly 5 short exercises with real-world based questions. "
        "Final response must be plain text only, no markdown or formatting."
    )
    reply = groq_task(system_prompt, combined, temperature=0.2, max_tokens=2500)
    logger.info("explain_more reply preview: %s", reply[:180])
    return jsonify({"reply": reply, "detail": True})

@app.route("/explain_solution", methods=["POST"])
def explain_solution():
    data = request.get_json(silent=True) or {}
    problem_text = session.get("last_problem", "")
    solution_text = (data.get("solution") or "").strip()

    if not problem_text or not solution_text:
        return jsonify({"reply": "⚠️ No problem or solution found for explanation.", "detail": False})

    combined_text = f"Problem:\n{problem_text}\n\nSolution:\n{solution_text}"
    system_prompt = (
           "You are an expert tutor. Produce a concise, step-by-step explanation of the provided Solution. "
        "OUTPUT RULES:\n"
        "1) Return ONLY an ordered markdown list (1., 2., 3., ...). Each list item must be one step on its own line.\n"
        "2) Do NOT write paragraphs or extra commentary. Do NOT provide a different solution—explain the given one.\n"
        "3) If code is present, explain each key line/block as separate steps.\n"
        "4) Keep each step focused (aim for ~8–30 words).\n\n"
        "Example output:\n"
        "1. Identify the input and desired output.\n"
        "2. Parse input and initialize variables.\n"
        "3. Loop over array to update max/min as needed.\n\n"
        "Return nothing else besides the ordered list."
    )
    reply = groq_task(system_prompt, combined_text, temperature=0.2, max_tokens=2500)
    return jsonify({"reply": reply, "detail": True})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = (request.form.get("message") or "").strip()
        user_lower = user_message.lower()

        # Handle direct greetings and common phrases without an API call
        if user_lower in ["hi", "hello", "hey", "greetings"]:
            return jsonify({"reply": "Hello! How can I help you today?", "detail": False})
        if "thank you" in user_lower or "thanks" in user_lower:
            return jsonify({"reply": "You're welcome!", "detail": False})

        uploaded = "file" in request.files and getattr(request.files["file"], "filename", "")

        file_text = ""
        if uploaded:
            f = request.files["file"]
            if f.filename.lower().endswith(".pdf"):
                try:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            file_text += page_text + "\n"
                except Exception:
                    f.seek(0)
                    images = convert_from_bytes(f.read(), dpi=300)
                    for img in images:
                        file_text += pytesseract.image_to_string(img) + "\n"
            else:
                img = Image.open(io.BytesIO(f.read()))
                file_text = pytesseract.image_to_string(img)
            
            if file_text.strip():
                session["last_file_text"] = file_text.strip()
                session.pop("last_problem", None)  # Clear last problem if a new file is uploaded
            
            if not user_message:
                text_to_sum = session.get("last_file_text", "")
                if not text_to_sum:
                    return jsonify({"reply": "⚠️ No text extracted from the file.", "detail": False})
                result = groq_chat(f"Summarize into 5–7 bullets:\n{text_to_sum}")
                return jsonify({"reply": result, "detail": True})

        combined_text = user_message
        if session.get("last_file_text"):
            combined_text += f"\n\n[TEXT FROM FILE]\n{session['last_file_text']}"
            
        # Check for problem-solving intent
        if "solve" in user_lower or "problem" in user_lower:
            coding_keywords = ['code', 'algorithm', 'function', 'program', 'python', 'javascript']
            is_coding_problem = any(kw in user_lower for kw in coding_keywords)
            
            session.pop("problem_hints", None)
            session.pop("problem_solution", None)
            session.pop("hint_count", None)

            if is_coding_problem:
                problem_data = generate_coding_problem_response(combined_text)
            else:
                problem_data = generate_hints_and_solution(combined_text)

            if problem_data:
                session["problem_hints"] = problem_data.get("hints", [])
                session["problem_solution"] = problem_data.get("solution", "Solution not available.")
                session["hint_count"] = 0
                session["last_problem"] = combined_text
                return jsonify({
                    "reply": "I've detected a problem. Do you want a hint or the full solution?",
                    "is_problem": True
                })

        # Check for specific "explain more" keywords
        options = detect_options(user_message)
        if options["want_explain"]:
            # If the user explicitly asks for an explanation, use the detailed task
            result = groq_task(
                system_prompt="You are an expert teacher. Provide a clear, step-by-step explanation using markdown.",
                user_prompt=f"Explain in detail: {combined_text}"
            )
            return jsonify({"reply": result, "detail": True})

        # If it's a general question or statement, use the conversational chat model
        else:
            result = groq_chat(f"Summarize or answer the following: {combined_text}")
            return jsonify({"reply": result, "detail": True})

    except Exception as e:
        logger.exception("chat route error")
        return jsonify({"reply": f"⚠️ Server error: {e}"}), 500

@app.route("/solve_problem", methods=["POST"])
def solve_problem():
    solution = session.get("problem_solution", "No solution available. Please provide a problem first.")
    session.pop("problem_hints", None)
    session.pop("hint_count", None)
    return jsonify({"reply": f"**Solution:**\n\n{solution}", "is_solution": True, "solution_text": solution})

@app.route("/get_hint", methods=["POST"])
def get_hint():
    hints = session.get("problem_hints", [])
    hint_count = session.get("hint_count", 0)
    max_hints = len(hints)

    if not hints:
        return jsonify({"reply": "No problem detected to provide a hint.", "is_hint": False})

    if hint_count >= max_hints:
        session.pop("problem_hints", None)
        session.pop("problem_solution", None)
        session.pop("hint_count", None)
        return jsonify({
            "reply": "Hint limit exceeded. Here is the solution.",
            "is_solution": True,
            "solution_text": session.get("problem_solution", "Solution not available.")
        })

    current_hint = hints[hint_count]
    session["hint_count"] = hint_count + 1

    return jsonify({
        "reply": f"**Hint {hint_count + 1}/{max_hints}:**\n\n{current_hint}",
        "is_hint": True,
        "hint_number": hint_count + 1,
        "total_hints": max_hints
    })

@app.route("/summarize", methods=["POST"])
def summarize():
    text = session.get("last_file_text", "")
    if not text:
        return jsonify({"reply": "⚠️ No text available for summarization.", "detail": False})
    result = groq_chat(f"Summarize into 5–7 bullets:\n{text}")
    return jsonify({"reply": result, "detail": True})

@app.route("/flashcards", methods=["POST"])
def flashcards():
    text = session.get("last_file_text", "")
    if not text:
        return jsonify({"cards": []})
    system_prompt = (
        "You are a flashcard generator. From the text, create possible Q&A flashcards. "
        "Keep questions short, answers concise and up to 6 lines long. Useful for revision. "
        "Don't create any extra text other than answers."
    )
    cards_text = groq_task(system_prompt, text, max_tokens=2000)
    cards = []
    for block in cards_text.split("Q:"):
        if "A:" in block:
            q_part, a_part = block.split("A:", 1)
            q_clean = q_part.strip().replace("\n", " ")
            a_clean = a_part.strip().replace("\n", " ")
            if q_clean and a_clean:
                cards.append({"q": q_clean, "a": a_clean})
    return jsonify({"cards": cards})

@app.route("/visualize", methods=["POST"])
def visualize():
    text = session.get("last_file_text", "")
    if not text:
        return jsonify({"reply": "⚠️ No text available for visualization."})

    system_prompt = (
        "You are an expert at creating diagrams from text. "
        "Your task is to analyze the user-provided content and generate a diagram in Graphviz DOT language syntax. "
        "The diagram should be a directed graph (digraph) or an undirected graph (graph). "
        "Return ONLY the DOT code block as plain text, without any introductory or concluding sentences. "
        "Ensure the DOT syntax is perfectly correct and will render without errors. "
        "Do not include markdown fences like '```' in the output. Just the raw code."
    )
    user_prompt = f"Generate a flowchart in DOT language from the following text:\n\n{text}"

    try:
        dot_code = groq_task(system_prompt, user_prompt, max_tokens=2000)
        clean_code = re.sub(r'```[a-z]*|```', '', dot_code, flags=re.DOTALL).strip()

        if not (clean_code.startswith("digraph") or clean_code.startswith("graph")):
            return jsonify({"reply": "⚠️ Failed to generate a valid Graphviz diagram. The AI produced an incorrect format."})

        kroki_url = "https://kroki.io/graphviz/svg"
        headers = {"Content-Type": "text/plain"}
        r = requests.post(kroki_url, data=clean_code.encode("utf-8"), headers=headers, timeout=20)
        r.raise_for_status()

        svg_bytes = r.content
        data_url = "data:image/svg+xml;base64," + base64.b64encode(svg_bytes).decode("ascii")
        return jsonify({"image_url": data_url})

    except requests.exceptions.RequestException as e:
        logger.exception("Kroki API error")
        return jsonify({"reply": f"⚠️ Kroki API error: {e}"})
    except Exception as e:
        logger.exception("visualize route error")
        return jsonify({"reply": f"⚠️ An unexpected error occurred: {e}"})

@app.route("/dictionary", methods=["POST"])
def dictionary():
    text = session.get("last_file_text", "")
    if not text:
        return jsonify({"reply": "⚠️ No text available for dictionary."})
    system_prompt = (
        "You are a simple dictionary. Extract the 8–12 most important technical terms "
        "from the text. For each term, explain simply with a tiny example."
    )
    result = groq_task(system_prompt, text, max_tokens=2000)
    return jsonify({"reply": result})

@app.route("/quiz", methods=["POST"])
def quiz():
    text = session.get("last_file_text", "")
    if not text:
        return jsonify({"quiz": None, "reply": "⚠️ No text available to generate a quiz."})
    system_prompt = (
        "You are an expert quiz generator. Create a comprehensive multiple-choice quiz from the text with a reasonable number of unique questions (5-10 questions). "
        "For each question, provide exactly four options (A, B, C, D), specify the correct one, and include a concise explanation for the correct answer. "
        "For incorrect answers, provide a brief explanation and suggest specific topics or concepts to review to gain a better understanding. "
        "The suggestions should be a list of actionable topics to cover for each wrong answer. "
        "Do not repeat questions. "
        "Format the output strictly as a JSON object with a 'quiz' key. "
        "Each item in the quiz list should have a 'question', an 'options' key (list of four strings), a 'correctAnswer' (A, B, C, or D), a 'correctExplanation' string, and a 'topicsToFocusOn' key which is a list of strings for topics."
    )
    user_prompt = f"Generate a quiz from the following text:\n\n{text}"
    try:
        quiz_response = groq_task(system_prompt, user_prompt, max_tokens=2000)
        quiz_response = re.sub(r"```[a-zA-Z]*", "", quiz_response).replace("```", "").strip()
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', quiz_response)

        if not json_match:
            return jsonify({"quiz": None, "reply": "⚠️ No valid JSON found in AI response."})
        quiz_json_string = json_match.group(0)
        try:
            quiz_data = safe_json_loads(quiz_json_string)

        except json.JSONDecodeError:
            return jsonify({"quiz": None, "reply": "⚠️ Quiz generation failed. Invalid JSON."})
        
        # Ensure all questions have a correct answer letter and topicsToFocusOn
        for q in quiz_data.get("quiz", []):
            if "answer" in q and "correctAnswer" not in q:
                q["correctAnswer"] = q.pop("answer")
            if q.get("correctAnswer") not in ["A", "B", "C", "D"]:
                logger.warning("AI produced a quiz with an invalid correct answer. Fixing it.")
                q["correctAnswer"] = q.get("options", [""])[0]  # Default to the first option
            if "topicsToFocusOn" not in q:
                q["topicsToFocusOn"] = []
                
        if "quiz" in quiz_data:
            return jsonify(quiz_data)
        return jsonify({"quiz": None, "reply": "⚠️ Failed to parse quiz from response."})
    except Exception as e:
        logger.exception("quiz route error")
        return jsonify({"quiz": None, "reply": f"⚠️ An unexpected error occurred: {e}"})
def safe_json_loads(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Step 1: Basic cleanup
        text = text.strip()
        text = text.replace("\r", "").replace("\n", " ")
        
        # Step 2: Replace curly quotes with normal quotes
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        
        # Step 3: Ensure double quotes for JSON keys/strings
        text = re.sub(r"'", '"', text)
        
        # Step 4: Remove trailing commas
        text = re.sub(r",\s*([}\]])", r"\1", text)
        
        # Step 5: Only keep JSON object/array
        match = re.search(r'(\{.*\}|\[.*\])', text)
        if match:
            text = match.group(0)
        
        return json.loads(text)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)