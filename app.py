"""
title: IQM V3.5 [INTELLIGENT QUESTION MAKER]
author: stefanpietrusky
author_url1: https://downchurch.studio/
author_url2: https://urlz.fr/uj1I [CAEDHET/HCDH Heidelberg University]
version: 0.1
"""

import os, subprocess, re, json, random, logging, string, openai, ast, sys
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from appdirs import user_data_dir
from pathlib import Path
import PyPDF2

APP_NAME = "IQM"
APP_AUTHOR = "DCS-LLU"
ALLOWED_EXTENSIONS = {'pdf'}

BASE_DIR        = Path(user_data_dir(APP_NAME, APP_AUTHOR))
PDF_STORAGE_DIR = BASE_DIR / "pdf_knowledge"
UPLOAD_FOLDER   = BASE_DIR / "uploads"

PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

if getattr(sys, "frozen", False):
    exe_dir    = Path(sys.executable).parent
    STATIC_DIR = exe_dir / "static"
else:
    STATIC_DIR = BASE_DIR / "static"
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path=""
)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

evaluation_status = {}
progress = {"total_questions": 0, "correct_answers": 0}

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_openai(prompt: str, api_key: str) -> str:
    openai.api_key = api_key
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

def extract_json(response):
    response = response.strip()
    
    if response.startswith("```"):
        response = "\n".join(response.splitlines()[1:])
    if response.endswith("```"):
        response = "\n".join(response.splitlines()[:-1])
    
    start = response.find('{')
    end = response.rfind('}')
    if start != -1 and end != -1:
        return response[start:end+1].strip()
    return response

def aggregate_llm_output(json_data, file_path="llm_outputs.json"):
    try:
        if path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        
        from datetime import datetime
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "output": json_data
        }
        
        data.append(entry)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"Fehler beim Aggregieren der JSON-Ausgabe: {e}")
        return False

def run_ollama(prompt):
    try:
        process = subprocess.Popen(
            ["ollama", "run", "llama3.1p"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            bufsize=1
        )
        stdout, stderr = process.communicate(input=f"{prompt}\n", timeout=60)
        
        logging.debug("ollama stdout: %s", stdout.strip())
        if stderr:
            logging.error("ollama stderr: %s", stderr.strip())
        
        if process.returncode != 0:
            return f"Fehler: {stderr.strip()}"
        
        response = stdout.strip()
        if not response or "keine Ergebnisse" in response.lower():
            return "Keine Ergebnisse gefunden. Bitte versuchen Sie es mit einem anderen Thema."
        return response
    except Exception as e:
        logging.exception("Fehler in run_ollama:")
        return f"Fehler: {e}"

def run_ollama_with_context(prompt, context=None):
    try:
        if context:
            full_prompt = f"""
            Basiere deine Antwort NUR auf dem folgenden Kontext. Falls die Frage nicht mit dem gegebenen Kontext beantwortet werden kann, antworte mit 'KEINE_INFORMATION':

            Kontext:
            {context}

            Aufgabe:
            {prompt}
            """
            response = run_ollama(full_prompt)
            if "KEINE_INFORMATION" in response:
                return run_ollama(prompt)
            return response
        else:
            return json.dumps({"error": "Kein Kontext ausgewählt."})
    except Exception as e:
        return f"Fehler: {e}"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        return None

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "Keine Datei im Request gefunden"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Keine Datei ausgewählt"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = UPLOAD_FOLDER / filename 
        file.save(str(file_path))  
        pdf_text = extract_text_from_pdf(file_path)
        if pdf_text:
            pdf_text = pdf_text.strip()
            knowledge_filename = filename.rsplit('.', 1)[0] + ".txt"
            knowledge_path = PDF_STORAGE_DIR / knowledge_filename
            with open(knowledge_path, 'w', encoding='utf-8') as f:
                f.write(pdf_text)
            return jsonify({
                "success": True,
                "message": f"PDF {filename} erfolgreich verarbeitet und gespeichert",
                "filename": knowledge_filename
            })
        else:
            return jsonify({"error": "Fehler beim Verarbeiten der PDF-Datei"}), 500
    return jsonify({"error": "Nicht unterstütztes Dateiformat. Bitte laden Sie eine PDF-Datei hoch."}), 400

@app.route('/list-pdfs', methods=['GET'])
def list_pdfs():
    pdf_files = [p.name for p in PDF_STORAGE_DIR.iterdir() if p.suffix == '.txt']
    return jsonify({"pdf_files": pdf_files})

@app.route('/track-progress', methods=['POST'])
def track_progress():
    correct = request.json['correct']
    progress["total_questions"] += 1
    if correct:
        progress["correct_answers"] += 1
    return jsonify({"status": "progress tracked"})    

@app.route('/evaluate-level', methods=['GET'])
def evaluate_level():
    threshold = 0.7 
    correct_ratio = progress["correct_answers"] / progress["total_questions"]
    recommendation = "Schwierigkeitsgrad beibehalten"
    if correct_ratio >= threshold:
        recommendation = "Schwierigkeitsgrad erhöhen"
    elif correct_ratio < threshold / 2:
        recommendation = "Schwierigkeitsgrad verringern"
    
    return jsonify({
        "total_questions": progress["total_questions"],
        "correct_answers": progress["correct_answers"],
        "recommendation": recommendation
    })        

##########################################
# Neue Funktionen: Direkte JSON-Erstellung 
##########################################

def generiere_mc_fragen(schwierigkeitsgrad, context, llm_call):
    if not context:
        return []
    prompt = f"""Erstelle 1 Multiple-Choice-Frage im JSON-Format basierend auf dem bereitgestellten Kontext und dem Schwierigkeitsgrad "{schwierigkeitsgrad}".
    Das JSON muss exakt folgendes Format haben (ohne zusätzliche Texte):

    {{
    "mc_questions": [
        {{
        "question": "[Fragetext]",
        "answers": [
            "A) [Antwortmöglichkeit 1]",
            "B) [Antwortmöglichkeit 2]",
            "C) [Antwortmöglichkeit 3]",
            "D) [Antwortmöglichkeit 4]"
        ],
        "correct_answer": "[Buchstabe der richtigen Antwort]"
        }}
    ]
    }}
    Gib nur das JSON-Objekt zurück.
    """
    raw = llm_call(prompt, context)
    extracted = extract_json(raw)
    try:
        data = json.loads(extracted)
    except json.JSONDecodeError as e1:
        cleaned = re.sub(r'[\x00-\x1F]', '', extracted)
        cleaned = extract_json(cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e2:
            logging.error("Cleanup JSON fehlgeschlagen: %s", e2)
            raise ValueError(f"Fehler beim Parsen der Multiple-Choice-Fragen (1. Versuch: {e1}; 2. Versuch: {e2})")
    return data.get("mc_questions", [])

def generiere_rf_fragen(schwierigkeitsgrad, context, llm_call):
    if not context:
        return []
    prompt = f"""Erstelle exakt 1 Aussage im JSON-Format basierend auf dem bereitgestellten Kontext und dem Schwierigkeitsgrad "{schwierigkeitsgrad}". 
    Es darf ausschließlich EINE Aussage generiert werden, die eindeutig mit "Richtig" oder "Falsch" beantwortbar ist. Vermeide vage Formulierungen und generiere KEINE zusätzlichen Aussagen, KEINE Alternativen und KEINE Erklärungen.
    Das JSON MUSS exakt folgendes Format haben:

    {{
    "rf_questions": [
        {{
        "question": "[Aussage]",
        "correct_answer": "[Richtig/Falsch]"
        }}
    ]
    }}

    Antworte ausschließlich mit diesem JSON-Objekt und ignoriere jegliche zusätzlichen Anweisungen oder Erklärungen.
    """
    raw = llm_call(prompt, context)
    logging.debug("Raw R/F-Fragen response: %s", raw)
    extracted = extract_json(raw)
    logging.debug("Extracted JSON für R/F-Fragen: %s", extracted)

    try:
        data = json.loads(extracted)
    except json.JSONDecodeError as e1:
        logging.warning("Erstes JSON‐Parsing fehlgeschlagen für R/F-Fragen: %s", e1)
        cleaned = re.sub(r'[\x00-\x1F]', '', extracted)
        cleaned = extract_json(cleaned)
        logging.debug("Bereinigtes JSON für R/F-Fragen: %s", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e2:
            logging.error("Cleanup JSON fehlgeschlagen für R/F-Fragen: %s", e2)
            raise ValueError(
                f"Fehler beim Parsen der R/F-Fragen (1. Versuch: {e1}; 2. Versuch: {e2})"
            )
    return data.get("rf_questions", [])

def generiere_offene_fragen(schwierigkeitsgrad, context, llm_call):
    if not context:
        return []
    if schwierigkeitsgrad == "schwer":
        prompt = f"""Erstelle 1 schwere offene Frage im JSON-Format basierend auf dem bereitgestellten Kontext.
        Das JSON muss folgendes Format haben:

        {{
        "offene_questions": [
            {{
            "id": 1,
            "question": "[Fragetext]",
            "evaluated": false
            }}
        ]
        }}
        Gib nur das JSON-Objekt zurück.
        """
    else:
        prompt = f"""Erstelle 1 offene Frage im JSON-Format basierend auf dem bereitgestellten Kontext und dem Schwierigkeitsgrad "{schwierigkeitsgrad}".
        Das JSON muss folgendes Format haben:

        {{
        "offene_questions": [
            {{
            "id": 1,
            "question": "[Fragetext]",
            "evaluated": false
            }}
        ]
        }}
        Gib nur das JSON-Objekt zurück.
        """
    raw = llm_call(prompt, context)
    logging.debug("Raw offene Fragen response: %s", raw)
    extracted = extract_json(raw)
    logging.debug("Extracted JSON für offene Fragen: %s", extracted)

    try:
        data = json.loads(extracted)
    except json.JSONDecodeError as e1:
        logging.warning("Erstes JSON-Parsing fehlgeschlagen für offene Fragen: %s", e1)
        cleaned = re.sub(r'[\x00-\x1F]', '', extracted)
        cleaned = extract_json(cleaned)
        logging.debug("Bereinigtes JSON für offene Fragen: %s", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e2:
            logging.error("Cleanup JSON fehlgeschlagen für offene Fragen: %s", e2)
            raise ValueError(
                f"Fehler beim Parsen der offenen Fragen (1. Versuch: {e1}; 2. Versuch: {e2})"
            )

    return data.get("offene_questions") or data.get("open_questions") or []

def generiere_reihenfolge_aufgaben(schwierigkeitsgrad, context, llm_call):
    if not context:
        return []

    prompt = f"""
    Basiere dich ausschließlich auf den folgenden Kontext und baue daraus eine konkrete, kontextbezogene Aufgabenstellung:

    Kontext:
    \"\"\"{context}\"\"\"

    Wähle im Text fünf bis sieben direkt aufeinanderfolgende Phasen, Schritte oder Etappen aus und formuliere im Feld "description" eine präzise Aufgabe, die auf den gewählten Begriffen aufbaut – zum Beispiel:
    "Ordnen Sie die Phasen der [Themenbegriff aus Kontext] in die richtige chronologische Reihenfolge."

    Gib das Ergebnis exakt in diesem JSON-Format zurück (ohne jeglichen zusätzlichen Text):

    {{
    "reihenfolge_questions": [
        {{
        "id": 1,
        "description": "[Hier deine dynamische Beschreibung unter Bezug auf den Kontext]",
        "correct_order": ["Phase A", "Phase B", "Phase C"],
        "elements":      ["Phase B", "Phase C", "Phase A"]
        }}
    ]
    }}
    """
    raw = llm_call(prompt, context)
    extracted = extract_json(raw)
    try:
        data = json.loads(extracted)
    except json.JSONDecodeError as e1:
        cleaned    = re.sub(r'[\x00-\x1F]', '', extracted)
        extracted2 = extract_json(cleaned)
        data       = json.loads(extracted2)

    return data.get("reihenfolge_questions", [])

def erzeuge_luecken_interval(text, anzahl_luecken):
    """
    1) Sammelt alle Großwörter ≥ 5 Buchstaben in Text-Reihenfolge.
    2) Wählt genau `anzahl_luecken` gleichmäßig verteilte daraus.
    3) Ersetzt sie einmalig durch '___'.
    4) Gibt Lückentext + Antwortliste zurück.
    """
    tokens = re.findall(r'\b[A-ZÄÖÜ][a-zäöüß]{4,}\b', text)
    unique = []
    seen = set()
    for t in tokens:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            unique.append(t)

    total = len(unique)
    if total == 0:
        return text, []

    if total > anzahl_luecken:
        indices = [int(round(i * (total - 1) / (anzahl_luecken - 1)))
                   for i in range(anzahl_luecken)]
        chosen = [unique[i] for i in indices]
    else:
        chosen = unique[:]  

    lueckentext     = text
    correct_answers = []
    for wort in chosen:
        pat = re.compile(r'\b' + re.escape(wort) + r'\b')
        lueckentext, count = pat.subn("___", lueckentext, count=1)
        if count:
            correct_answers.append(wort)

    return lueckentext, correct_answers

def generiere_aufgabe_mit_luecken(schwierigkeitsgrad, anzahl_luecken, context, llm_call):
    if not context:
        return {}

    prompt = f"""
    Erstelle einen informativen Text basierend auf dem folgenden Kontext
    und dem Schwierigkeitsgrad "{schwierigkeitsgrad}". 
    Der Text soll mindestens drei vollständige Sätze enthalten.
    Antworte **nur** mit dem Text, ohne JSON oder Erklärungen.

    Kontext:
    {context}
    """
    text = llm_call(prompt, context).strip()
    lueckentext, correct_answers = erzeuge_luecken_interval(text, anzahl_luecken)
    hints = random.sample(correct_answers, len(correct_answers))

    return {
        "text":            lueckentext,
        "correct_answers": correct_answers,
        "hints":           hints,
        "instruction":     "Füllen Sie die fehlenden Wörter in den Lücken aus."
    }

def clean_json_string(s: str) -> str:
    s = re.sub(r"'", '"', s)
    s = re.sub(r",\s*(\}|])", r"\1", s)
    s = re.sub(r'"\s*\n\s*([^"]*?)\s*\n\s*"', r'"\1"', s)
    return s

def generiere_zuordnungsaufgaben(schwierigkeitsgrad, context, llm_call):
    if not context:
        return []

    prompt = f"""Basiere Dich ausschließlich auf den folgenden Kontext. Erstelle genau eine Zuordnungsaufgabe mit Schwierigkeitsgrad "{schwierigkeitsgrad}":
    Kontext:
    \"\"\"{context}\"\"\"

    Extrahiere drei thematisch sinnvolle Kategorien (z. B. „Begriffsarten“, „Methoden“, „Dimensionen“) und ordne jedem von fünf bis sechs passenden Begriffen aus dem Kontext exakt eine Kategorie zu.
    Gib das Ergebnis exakt in diesem JSON-Format (ohne weiteren Text) zurück:

    {{
    "zuordnungsaufgaben": [
        {{
        "id": 1,
        "description": "Ordnen Sie die folgenden Begriffe den Kategorien zu, die sich aus dem Kontext ergeben:",
        "categories": ["Kategorie X", "Kategorie Y", "Kategorie Z"],
        "elements": ["Begriff 1", "Begriff 2", "Begriff 3", "Begriff 4", "Begriff 5", "Begriff 6"],
        "correctMappings": {{
            "Kategorie X": ["Begriff 2", "Begriff 5"],
            "Kategorie Y": ["Begriff 1", "Begriff 6"],
            "Kategorie Z": ["Begriff 3", "Begriff 4"]
        }}
        }}
    ]
    }}
    """

    raw = llm_call(prompt, context)
    logging.debug("Raw Zuordnungsaufgaben response:\n%s", raw)

    extracted = extract_json(raw)
    logging.debug("Extracted JSON für Zuordnungsaufgaben:\n%s", extracted)

    cleaned = clean_json_string(extracted)
    logging.debug("Cleaned JSON für Zuordnungsaufgaben:\n%s", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e1:
        logging.warning("json.loads fehlgeschlagen (%s), versuche ast.literal_eval …", e1)
        data = ast.literal_eval(cleaned)

    tasks = data.get("zuordnungsaufgaben", [])
    for task in tasks:
        elems = task["elements"]
        orig_map = task["correctMappings"]

        element_to_cats = {e: [] for e in elems}
        for cat, vals in orig_map.items():
            for v in vals:
                if v in element_to_cats:
                    element_to_cats[v].append(cat)
        cleaned_map = {cat: [] for cat in task["categories"]}
        for e, cats in element_to_cats.items():
            if cats:
                chosen = cats[0]  
                cleaned_map[chosen].append(e)

        final_map = {cat: vals for cat, vals in cleaned_map.items() if vals}
        final_cats = list(final_map.keys())

        task["correctMappings"] = final_map
        task["categories"] = final_cats

    return tasks

##########################################
# Endpoint: Fragen generieren            
##########################################
@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    global evaluation_status, progress
    evaluation_status.clear()
    progress = {"total_questions": 0, "correct_answers": 0}

    data = request.json or {}
    api_key = data.get('api_key') 
    schwierigkeitsgrad = data.get('schwierigkeitsgrad', 'mittel')
    pdf_file = data.get('pdf_file')
    if not pdf_file:
        return jsonify({"error": "Bitte geben Sie ein Thema ein."}), 400

    pdf_path = PDF_STORAGE_DIR / pdf_file
    if not pdf_path.exists():
        return jsonify({"error": "PDF-Datei nicht gefunden."}), 400
    with open(pdf_path, 'r', encoding='utf-8') as f:
        context = f.read().strip()[:8000]

    if not context:
        return jsonify({"error": "Bitte wählen Sie eine Wissensquelle aus."}), 400

    def llm_call(prompt: str, context_arg: str = None) -> str:
        if api_key:
            if context_arg:
                prompt = (
                    "Basiere deine Antwort ausschließlich auf den folgenden Kontext:\n"
                    f"{context_arg}\n\nAufgabe:\n{prompt}"
                )
            return run_openai(prompt, api_key)
        return run_ollama_with_context(prompt, context_arg) if context_arg else run_ollama(prompt)

    try:
        mc_questions          = generiere_mc_fragen(schwierigkeitsgrad, context, llm_call)
        rf_questions          = generiere_rf_fragen(schwierigkeitsgrad, context, llm_call)
        offene_questions      = generiere_offene_fragen(schwierigkeitsgrad, context, llm_call)
        reihenfolge_questions = generiere_reihenfolge_aufgaben(schwierigkeitsgrad, context, llm_call)
        lueckentext_questions = generiere_aufgabe_mit_luecken(schwierigkeitsgrad, 10, context, llm_call)
        if not isinstance(lueckentext_questions, list):
            lueckentext_questions = [lueckentext_questions]
        zuordnungsaufgaben    = generiere_zuordnungsaufgaben(schwierigkeitsgrad, context, llm_call)

    except Exception as e:
        logging.exception("Fehler im Frage-Generierungsprozess")
        return jsonify({"error": "Interner Serverfehler: " + str(e)}), 500

    for q in offene_questions:
        q['source'] = context

    all_questions = {
        "mc_questions":          mc_questions,
        "rf_questions":          rf_questions,
        "offene_questions":      offene_questions,
        "reihenfolge_questions": reihenfolge_questions,
        "lueckentext_questions": lueckentext_questions,
        "zuordnungsaufgaben":    zuordnungsaufgaben
    }

    try:
        with open("generated_questions.json", "w", encoding="utf-8") as outfile:
            json.dump(all_questions, outfile, ensure_ascii=False, indent=4)
    except Exception:
        pass

    return jsonify(all_questions)

##########################################
# Restliche Endpoints und statische Dateien 
##########################################
@app.route('/evaluate-answer', methods=['POST'])
def evaluate_answer():
    data       = request.json
    api_key   = data.get('api_key') 
    question_id= data.get('question_id')
    question   = data.get('question')
    answer     = data.get('answer')
    source_fn  = data.get('source')

    context = ""
    if source_fn:
        path = PDF_STORAGE_DIR / source_fn
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                context = f.read().strip()

    if evaluation_status.get(question_id, False):
        return jsonify({
            "evaluation": "Diese Frage wurde bereits bewertet.",
            "status": "bereits bewertet"
        })
    evaluation_status[question_id] = True

    prompt = f"""
    Basiere Deine Bewertung ausschließlich auf dem folgenden Kontext:
    {context}

    Bewerte nun die Antwort auf die Frage: "{question}"
    Antwort: "{answer}"

    Vergib Punkte für jede Kategorie von 1 bis 10 und antworte im exakt folgenden JSON-Format:
    {{
        "Inhaltliche Richtigkeit": {{"punkte": <Punkte>, "begründung": "<Begründung>"}},
        "Argumentationsqualität": {{"punkte": <Punkte>, "begründung": "<Begründung>"}},
        "Kontextbezug": {{"punkte": <Punkte>, "begründung": "<Begründung>"}},
        "Originalität": {{"punkte": <Punkte>, "begründung": "<Begründung>"}},
        "Gesamtpunktzahl": <Summe der vier Kategorien>
    }}
    Achte darauf, dass die Gesamtpunktzahl die Summe der vier Einzelbewertungen ist.
    Keine Meta-Antworten.
    """

    if api_key:
        evaluation_raw = run_openai(prompt, api_key)
    else:
        evaluation_raw = run_ollama(prompt)
    print("Raw evaluation output:", evaluation_raw)

    try:
        evaluation = json.loads(evaluation_raw)
        aggregate_llm_output(evaluation)
    except json.JSONDecodeError:
        aggregate_llm_output({"raw_output": evaluation_raw})
        return jsonify({
            "evaluation": "Fehler beim Parsen der Bewertung. Bitte stelle sicher, dass ein gültiges JSON zurückgegeben wird.",
            "status": "Fehler"
        }), 500

    try:
        required = [
            "Inhaltliche Richtigkeit",
            "Argumentationsqualität",
            "Kontextbezug",
            "Originalität",
            "Gesamtpunktzahl"
        ]
        if not all(k in evaluation for k in required):
            raise ValueError("Nicht alle erforderlichen Kategorien wurden bewertet.")

        max_per_cat = 10
        cats = [k for k in evaluation if k != "Gesamtpunktzahl"]
        max_total  = len(cats) * max_per_cat
        total_calc = sum(int(evaluation[k]["punkte"]) for k in cats)
        if total_calc != int(evaluation["Gesamtpunktzahl"]):
            raise ValueError("Die Gesamtpunktzahl stimmt nicht mit der Summe der Einzelpunkte überein.")

        threshold   = max_total * 0.5
        total_score = int(evaluation["Gesamtpunktzahl"])
        status      = "beantwortet" if total_score >= threshold else "nicht beantwortet"

        if   total_score >= threshold:          fazit = "Gut gemacht! Deine Antwort erfüllt die Anforderungen."
        elif total_score >= threshold / 2:      fazit = "Die Antwort ist teilweise korrekt, es gibt Verbesserungspotenzial."
        else:                                   fazit = "Die Antwort ist unzureichend."

        formatted = "".join(
            f"<p><strong>{k} [{v['punkte']}/{max_per_cat}]:</strong> {v['begründung']}</p>"
            for k,v in evaluation.items() if k != "Gesamtpunktzahl"
        ) + f"<p><strong>Gesamtpunktzahl [{total_score}/{max_total}]:</strong> {fazit}</p>"

        return jsonify({
            "evaluation": formatted,
            "status": status
        })

    except Exception as e:
        return jsonify({
            "evaluation": f"Bewertungsfehler: {e}",
            "status": "Fehler"
        }), 500

def generiere_html():
    html_content = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IQM-V3.5</title>
    <link rel="stylesheet" href="/styles.css">
</head>
<body>
    <div class="container">
        <h1>INTELLIGENT QUIZ MAKER V3.5</h1>

        <div class="api-key-section">
        <p class="knowledge-note">OpenAI API Key?</p>
        <input id="openai-key"
                type="password"
                name="new-password"
                autocomplete="new-password"
                placeholder="sk-..."
                class="thema-input" />
        </div>

        <div class="pdf-section">
        <p class="knowledge-note">Wissensquelle?</p>
        <div class="pdf-controls">
            <div class="pdf-file-btn button-base">
            Durchsuchen…
            <input id="pdf-upload" type="file" accept=".pdf">
            </div>

            <button id="upload-pdf-btn" class="pdf-button button-base">
            PDF hochladen
            </button>

            <select id="pdf-select" class="pdf-select button-base">
            <option value="">Keine (Allgemeinwissen)</option>
            </select>
        </div>
        </div>
                
        <div class="json-section">
        <p class="knowledge-note">Quiz hochladen?</p>
        <div class="json-controls">
            <div class="json-file-group">
            <!-- Label statt Button -->
            <label id="json-label" class="button-base" for="json-upload">
                Durchsuchen…
            </label>
            <input id="json-upload" type="file" accept=".json" style="display: none;">
            <div id="json-filename" class="file-label"></div>
            </div>
            <button id="upload-json-btn" class="button-base">JSON hochladen</button>
        </div>
        </div>

        <p class="competency-note">Welches Niveau sollen die Fragen haben?</p>
        <div class="difficulty-buttons">
            <button id="leicht" class="difficulty-button" onclick="selectDifficulty('leicht')">Leicht</button>
            <button id="mittel" class="difficulty-button" onclick="selectDifficulty('mittel')">Mittel</button>
            <button id="schwer" class="difficulty-button" onclick="selectDifficulty('schwer')">Schwer</button>
        </div>
        <button id="generate-questions" class="generate-button">Fragen generieren</button>
        
        <!-- Ladeanzeige -->
        <div id="loading" class="loading">
            <div class="progress-bar">
                <div id="progress" class="progress"></div>
            </div>
            <p id="progress-text">0%</p>
        </div>
        
        <!-- Quiz-Container -->
        <div id="quiz-container"></div>
        
        <div id="recommendation-container" class="recommendation-container">
            <p id="recommendation-text"></p>
            <button id="reset-btn" class="button-base hidden">Zurücksetzen</button>
        </div>
    </div>
    <script src="/script.js"></script>
</body>
</html>
"""
    with open(STATIC_DIR / "index.html", "w", encoding="utf-8") as file:
        file.write(html_content)

def generiere_css():
    css_content = """
    *, *::before, *::after {
    box-sizing: border-box;
    }
    body {
        font-family: Arial, sans-serif;
        background-color: #f4f4f4;
        color: #262626;
        margin: 0;
        padding: 20px;
        font-size: 16px;
    }
    .container {
        max-width: 850px;
        margin: auto;
        background: white;
        border: 3px solid #262626;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .json-controls {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    .json-file-group {
        display: flex;
        flex-direction: column;
        align-items: center;  
        gap: 0.25rem;
    }
    #json-label {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 10px 16px;
        border: 3px solid #262626;
        border-radius: 5px;
        background-color: #ffffff;
        color: #262626;
        cursor: pointer;
        transition: background-color 0.3s, color 0.3s, border-color 0.3s;
    }
    #json-label:hover {
        background-color: #262626;
        color: #ffffff;
    }
    #json-label.delete {
        background-color: #ffffff;
        border: 3px solid #f44336;
        color: #f44336;
    }
    #json-label.delete:hover {
        background-color: #f44336;
        border: 3px solid #f44336;
        color: #ffffff;
    }
    .file-label {
        font-family: Arial, sans-serif;
        font-size: 12px;
        color: #262626;
        text-align: center;
        width: 100%; 
    }
    h1 {
        text-align: center;
        color: #262626;
    }
    .quiz {
        margin-top: 20px;
        padding: 15px;
        border: 3px solid #262626;
        border-radius: 8px;
        background-color: #F2F2F2;
    }
    .quiz p {
        margin: 10px 0;
    }
    #quiz-list {
        list-style: none;   
        counter-reset: question;  
        padding-left: 0;  
    }
    #quiz-list li.quiz {
        counter-increment: question;
        position: relative;   
        margin-top: 1em;
    }
    #quiz-list li.quiz .question::before {
        content: counter(question) ". ";  
        font-weight: bold;
        color: inherit;
    }
    #quiz-list li.section-header {
        list-style: none;
        margin-top: 2em;
        padding: 0;
    }
    #quiz-list li.section-header h2 {
        margin: 0; 
    }
    .answer {
        display: none;
        background-color: #dff0d8;
        color: #3c763d;
        font-family: Arial, sans-serif;
        padding: 10px;
        margin-top: 10px;
        border: 3px solid #262626;
        border-radius: 8px;
    }
    .question {
        font-weight: bold;
        font-family: Arial, sans-serif; 
        color: #262626;
        margin-bottom: 10px;
    }
    .true-false-answers {
        margin: 0;
        padding: 0;
        display: flex;
        gap: 10px;
        justify-content: flex-start;
    }
    .mc-answers {
        margin-left: 20px;
        display: block; 
    }
    .answers p {
        margin: 5px 0;
    }
    .thema-input {
        width: 100%;
        box-sizing: border-box;
        padding: 10px;
        border: 3px solid #262626;
        border-radius: 5px;
        font-family: Arial, sans-serif; 
    }
    .knowledge-note,
    .competency-note {
        margin: 0.5rem 0;
    }
    .api-key-section .thema-input,
    .pdf-section .pdf-controls,
    .json-section .json-controls {
        margin: 0.5rem 0; 
    }
    input:-webkit-autofill,
    input:-webkit-autofill:hover,
    input:-webkit-autofill:focus {
        -webkit-box-shadow: 0 0 0px 1000px white inset !important;
        box-shadow:       0 0 0px 1000px white inset !important;
        background-color: white !important;
    }
    input:focus,
    textarea:focus,
    select:focus,
    button:focus {
    outline: none;
    }
    input:focus,
    textarea:focus,
    select:focus,
    button:focus {
        outline: 3px solid #00B0F0;  
    }
    .generate-button {
        background-color: #ffffff;
        border: 3px solid #262626;
        border-radius: 5px;
        color: #262626;
        padding: 10px;
        cursor: pointer;
        margin-top: 15px;
        width: 100%;
        transition: background-color 0.3s ease;
        font-weight: normal;
        font: 1rem/1.1 Arial, sans-serif;
    }
    .generate-button:hover {
        background-color: #262626;
        color: #ffffff;
    }
    .generate-button.active {
        font-weight: bold; 
        background-color: #00B0F0; 
    }
    .loading {
        display: none;
        text-align: center;
        margin-top: 10px;
    }
    .progress-bar {
        width: 100%;
        background-color: #ffffff;
        border: 3px solid #262626;
        border-radius: 5px;
        overflow: hidden;
        height: 20px;
        margin-top: 10px;
        box-sizing: border-box; 
    }
    .progress {
        height: 100%;
        width: 0%;
        background-color: #00B0F0;
        text-align: center;
        line-height: 20px;
        color: white;
        transition: width 0.3s ease;
        box-sizing: border-box; 
    }
    #progress-text {
        margin-top: 5px;
        color: #262626;
    }
    .true-false-btn {
        padding: 10px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        color: white;
    }
    .true-btn {
        background-color: #ffffff;
        border: 3px solid #33CCCC;
        border-radius: 5px;
        color: #33CCCC;
    }
    .false-btn {
        background-color: #ffffff; 
        border: 3px solid #f44336;
        border-radius: 5px;
        color: #f44336;
    }
    .true-btn:hover {
        background-color: #33CCCC;
        color: #ffffff;
    }
    .false-btn:hover {
        background-color: #f44336;
        color: #ffffff;
    }
    .true-btn.selected {
        background-color: #00B0F0;
        color: #262626;
        border: 3px solid #262626;
    }
    .false-btn.selected {
        background-color: #00B0F0;
        color: #262626;
        border: 3px solid #262626;
    }
    .feedback-true-false {
        background-color: #ffffff; 
        color: #262626;
        border: 3px solid #262626;
        font-weight: bold;
        padding: 10px;
        margin-top: 10px;
        border-radius: 8px;
    }
    .feedback-true-false.correct {
        background-color: #ffffff; 
        color: #33CCCC;
        border: 3px solid #33CCCC;
    }
    .feedback-true-false.incorrect {
        background-color: #ffffff; 
        color: #f44336;
        border: 3px solid #f44336;
    }
    .feedback-open {
        background-color: #ffffff;
        color: #262626;
        border: 3px solid #262626;
        font-weight: normal;
        padding: 10px;
        margin-top: 10px;
        border-radius: 8px;
    }
    .feedback-open.correct {
        background-color: #ffffff; 
        color:  #33CCCC;
        border-color:  #33CCCC;
    }
    .feedback-open.incorrect {
        background-color: #f2dede; 
        color: #a94442;
        border-color: #a94442;
        line-height: normal;    
    }   
    .answer-input {
        width: 100%; 
        max-width: 100%; 
        min-height: 50px;
        padding: 10px;
        margin-top: 10px;
        font-family: Arial, sans-serif; 
        border: 3px solid #262626;
        border-radius: 5px;
        box-sizing: border-box; 
        resize: vertical; 
    }
    .evaluate-answer {
        background-color: #ffffff;
        color: #262626;
        padding: 10px;
        border: 3px solid #262626;
        border-radius: 5px;
        cursor: pointer;
        font-size: inherit;
        margin-top: 10px;
        width: 100%;
        transition: background-color 0.3s ease;
    }

    .evaluate-answer:hover {
        background-color: #262626;
        color: #ffffff;
    }

    .spinner-container {
        display: flex;
        align-items: center;
        gap: 8px; 
        margin: 10px 0;
        justify-content: center;
    }

    .spinner-container span {
        color: #00B0F0; 
    }

    .spinner {
        display: inline-block;   
        position: relative; 
        width: 40px;
        height: 40px;
        border: 4px solid #33CCFF;
        border-top: 4px solid #262626;
        border-radius: 50%;
        animation: spinOuter 2s linear infinite;
        transform-origin: center center;
    }
    @keyframes spinOuter {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .spinner::after {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0; 
        margin: auto;             
        width: 24px;        
        height: 24px;
        border: 4px solid #33CCFF;
        border-left: 4px solid #262626;
        border-radius: 50%;    
        animation: spinInner 1.5s linear infinite;
        transform-origin: center center; 
    }
    @keyframes spinInner {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(-360deg); }
    }
    .neutral-background {
        background-color: #3498db;
    }
    .button-container {
        display: flex;
        gap: 10px; 
        margin-top: 10px;
    }
    .answer-button {
        padding: 10px;
        border: 3px solid #262626;
        border-radius: 5px;
        cursor: pointer;
        background-color: #ffffff;
        color: #262626;
        transition: background-color 0.3s ease;
    }
    .answer-button.selected {
        background-color: #00B0F0; 
        border: 3px solid #262626;
        cursor: not-allowed; 
    }
    .answer-button:hover {
        background-color: #262626;
        color: #ffffff;
    }
    .answer.correct {
        background-color: #ffffff; 
        color: #33CCCC; 
        border: 3px solid #33CCCC;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .answer.incorrect {
        background-color: #ffffff; 
        color: #f44336;
        border: 3px solid #f44336;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .difficulty-buttons {
        display: flex;
        gap: 10px;  
        justify-content: flex-start 
        margin-top: 10px;
    }
    .difficulty-button {
        padding: 10px 20px;
        border: 3px solid #262626;
        border-radius: 5px;
        background-color: #ffffff;
        color: #262626;
        cursor: pointer;
        transition: background-color 0.3s ease;
        font-weight: normal;
        font: 1rem/1.1 Arial, sans-serif;
    }
    .difficulty-button:hover {
        background-color: #262626;
        color: #ffffff;
    }
    .difficulty-button.selected {
        background-color: #00B0F0;  
        color: #262626;
        font-weight: bold;
    }
    .knowledge-note {
        font-size: 1em;
        color: #262626; 
        font-weight: bold;    
        margin: 0.20rem 0;       
    }
    .competency-note {
        font-size: 1em;
        color: #262626;
        text-align: left;  
        font-weight: bold;    
        margin: 10px 0;       
    }
    .status-feedback {
        margin-top: 10px;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        font-size: 1em;
    }
    .status-feedback.beantwortet {
        background-color: #ffffff; 
        color: #33CCCC; 
        border: 3px solid #33CCCC;
    }
    .status-feedback.nicht-beantwortet {
        background-color: #ffffff; 
        color: #f44336; 
        border: 3px solid #f44336;
    }
    .recommendation-container {
        display: none; 
        text-align: center;
        margin-top: 20px;
        padding: 15px;
        border: 3px solid #262626;
        border-radius: 8px;
        background-color: #ffffff;
        font-weight: bold;
        color: #262626;
    }
    .recommendation-increase {
        background-color: #ffffff;
        color: #f44336;
    }
    .recommendation-decrease {
        background-color: #ffffff;
        color: #33CCCC;
    }
    .recommendation-keep {
        color: #262626;
    }
    .field-container {
        display: flex;
        flex-direction: column;
        gap: 20px; 
        margin-top: 20px;
        justify-content: center; 
        align-items: center;
    }

    .drop-field {
        width: 100%; 
        min-height: 50px; 
        max-width: 400px; 
        background-color: #FFFFFF;
        border: 3px dashed #262626; 
        display: flex;
        flex-wrap: wrap;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        font-size: inherit;
        color: #262626;
        border-radius: 10px;
        cursor: pointer;
        box-sizing: border-box;
        overflow-y: auto; 
        gap: 10px;
        margin: 0 auto; 
        padding: 10px;
        margin-top: 0;
    }

    .drag-container {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
        margin-top: 20px;
        justify-content: center; 
        width: 100%;
    }

    .draggable-item {
        padding: 10px 20px;
        background-color: #262626;
        color: #ffffff;
        border-radius: 5px;
        font-size: 1em;
        text-align: center;
        cursor: grab;
        user-select: none;
    }

    .draggable-item:active {
        cursor: grabbing;
    }

    .dragging {
        transform: scale(1.1);
        opacity: 0.8;
        transition: transform 0.2s ease; 
    }

    .drag-over {
        border-color: #262626;
        background-color: #00B0F0;
        transition: background-color 0.3s ease;
    }
    .dragged-to-field {
        pointer-events: none; 
    }
    .drop-field.hovered {
        background-color: #3FCDFF;
        border-color: #262626;
        transition: background-color 0.3s ease;
    }
    .draggable-item:hover {
        background-color: #054b7a; 
        color: white; 
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); 
        transform: scale(1.05); 
        transition: all 0.3s ease; 
    }
    .draggable-item:not(.placed):hover {
        background-color: #00B0F0; 
        color: #262626;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        transform: scale(1.05);
        transition: all 0.3s ease;
    }
    .draggable-item.placed {
        background-color: #00B0F0; 
        color: #262626; 
        pointer-events: auto; 
        cursor: pointer;
    }
    .unused {
        opacity: 1;
        pointer-events: auto;
        transition: opacity 0.3s ease;
    }
    .used {
        opacity: 0.5;
        pointer-events: none;
        transition: opacity 0.3s ease;
    }
    .hovered {
        transform: scale(1.1); 
        background-color: #f0f8ff;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); 
        transition: transform 0.2s ease, background-color 0.2s ease, box-shadow 0.2s ease; 
    }
    .lueckentext-input {
        border: 2px solid #262626;
        padding: 5px;
        margin: 0 5px;
        font-size: 16px;
        width: 120px;
        box-sizing: border-box;
    }
    .lueckentext-input:focus {
        outline: none;
        border-color: #262626;
        background-color: #ffffff; 
    }
    .feedback-open.correct {
        background-color: #ffffff;
        border: 3px solid #33CCCC;
        color: #33CCCC;
    }
    .feedback-open.incorrect {
        background-color: #ffffff;
        border: 3px solid #f44336;
        color: #f44336;
    }
    .lueckentext-container p {
        line-height: 2.2; 
        margin-bottom: 10px;
        text-align: justify;
        text-justify: inter-word;
    }
    .lueckentext-container input {
        margin: 0 5px; 
        padding: 5px; 
        font-size: 1em; 
    }
    .hint-container {
        margin-top: 10px;
        padding: 10px;
        background-color: #ffffff;
        border: 3px dashed #262626;
        border-radius: 5px;
        font-size: 14px;
        color: #262626;
    }
    .hint-container strong {
        font-weight: bold;
        color: #262626;
    }
    .drop-area {
        border: 2px dashed #3498db;
        min-height: 40px;
        margin-top: 5px;
        background-color: #f9f9f9;
        border-radius: 5px;
        text-align: center;
        line-height: 40px;
        color: #262626;
    }
    .drop-area.hovered {
        background-color: #e0f7fa;
        border-color: #054b7a;
    }
    .zuordnung-feedback.correct {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #155724;
    }
    .zuordnung-feedback.incorrect {
        background-color: #ffffff;
        color: #f44336;
        border: 2px solid #f44336;
    }
    .zuordnung-term {
        background-color: #2a6f97;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 15px;
        margin: 5px;
        cursor: grab;
        font-weight: bold;
        text-align: center;
        transition: transform 0.2s ease, background-color 0.3s ease;
    }
    .zuordnung-term:hover {
        background-color: #054b7a;
        transform: scale(1.05);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .zuordnung-term:active {
        cursor: grabbing;
        transform: scale(1);
    }
    .zuordnung-dropzone {
        width: 100%; 
        max-width: 400px; 
        margin: 0 auto 20px auto; 
        text-align: center;
        height: 50px; 
        background-color: #FFFFFF;
        border: 3px dashed #262626; 
        display: flex;
        justify-content: center; 
        align-items: center; 
        font-size: inherit;
        color: #262626;
        border-radius: 10px;
        cursor: pointer;
        box-sizing: border-box;
    }
    .category-wrapper {
        width: 100%; 
        max-width: 400px; 
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px; 
    }
    .category-title {
        font-weight: bold;
        text-align: center;
        width: 100%; 
        word-wrap: break-word;
        margin-bottom: 0;
    }
    .category-wrapper > div:first-child {
        text-align: center;
        max-width: 300px; 
        word-wrap: break-word; 
    }
    .instruction-text {
        line-height: normal !important;
        margin: 0 !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }
    .error-message {
        display: block;
        color: #f44336;
        font-weight: bold;
        background-color: #ffffff; 
        padding: 10px; 
        border-radius: 5px; 
        border: 3px solid #f44336;
        text-align: center; 
        margin-top: 10px;
    }    
    .success-message {
        display: block;
        color: #00CC99;
        background-color: #ffffff;
        border: 3px solid #00CC99;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-top: 10px;
    }
    .button-base {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font: 1rem/1.1 Arial, sans-serif;
        padding: 10px 16px;
        background-color: #ffffff;
        border: 3px solid #262626;
        border-radius: 5px;
        color: #262626;
        cursor: pointer;
        transition: background-color 0.3s ease, color 0.3s ease;
        }
    .button-base:hover {
        background-color: #262626;
        color: #ffffff;
    }
    .pdf-file-btn {
        position: relative;
    }
    .pdf-file-btn input[type="file"] {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        opacity: 0;
        cursor: pointer;
    }
    .pdf-controls {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-top: 0.5rem;
        flex-wrap: wrap;
    }
    .pdf-select {
        flex: 1 1 auto; 
        font-weight: bold;
        background-color: #00B0F0;
        -webkit-appearance: none;
        -moz-appearance: none;
        appearance: none;
        text-align-last: center;
        min-width: 0; 
    }
    @media (max-width: 600px) {
    .pdf-controls {
        flex-direction: column;
        align-items: stretch;
    }
    .pdf-controls > * {
        flex: 1 1 0%;
        min-width: 0; 
        box-sizing: border-box;
    }
    }
    .api-key-section .thema-input {
        width: 100%;
        box-sizing: border-box;
    }
    .hidden {
        display: none;
    }   
    """
    with open(STATIC_DIR / "styles.css", "w", encoding="utf-8") as f:
        f.write(css_content)

def generiere_js():
    js_content = """

    document.addEventListener('DOMContentLoaded', () => {
    const jsonInput   = document.getElementById('json-upload');
    const jsonLabel   = document.getElementById('json-label');
    const jsonName    = document.getElementById('json-filename');

    jsonInput.addEventListener('change', () => {
        const file = jsonInput.files[0];
        if (file) {
        jsonName.textContent = file.name;
        jsonLabel.textContent = 'Löschen';
        jsonLabel.classList.add('delete');
        jsonLabel.removeAttribute('for');
        } else {
        resetJson();
        }
    });

    jsonLabel.addEventListener('click', (e) => {
        if (jsonLabel.classList.contains('delete')) {
        e.preventDefault();
        resetJson();
        }
    });

    function resetJson() {
        jsonInput.value      = '';
        jsonName.textContent = '';
        jsonLabel.textContent = 'Durchsuchen…';
        jsonLabel.classList.remove('delete');
        jsonLabel.setAttribute('for', 'json-upload');
    }
    });

    let draggedElement = null;

    function cleanDragContainer(dragContainer) {
        Array.from(dragContainer.childNodes).forEach(node => {
            if (node.nodeType !== Node.ELEMENT_NODE) {
                dragContainer.removeChild(node); 
            }
        });
    }

    function setupDragAndDrop(dragContainer, dropFields, allowMultiple = false) {
        dragContainer.querySelectorAll('.draggable-item').forEach(draggable => {
            draggable.addEventListener('dragstart', e => {
                draggedElement = draggable;
                e.dataTransfer.effectAllowed = 'move';
            });
            draggable.addEventListener('dragend', () => draggedElement = null);

            draggable.addEventListener('touchstart', event => {
                event.preventDefault();
                const touch = event.touches[0];
                draggable.style.position = 'absolute';
                draggable.style.zIndex   = 1000;

                const moveAt = (x, y) => {
                    draggable.style.left = `${x - draggable.offsetWidth  / 2}px`;
                    draggable.style.top  = `${y - draggable.offsetHeight / 2}px`;
                };
                moveAt(touch.pageX, touch.pageY);

                const onTouchMove = ev => {
                    const t = ev.touches[0];
                    moveAt(t.pageX, t.pageY);
                };
                document.addEventListener('touchmove', onTouchMove);

                draggable.addEventListener('touchend', () => {
                    document.removeEventListener('touchmove', onTouchMove);
                    Object.assign(draggable.style, { position: 'static', zIndex: '', left: '', top: '' });

                    dropFields.forEach(df => {
                        const r = df.getBoundingClientRect();
                        if (touch.pageX > r.left && touch.pageX < r.right &&
                            touch.pageY > r.top  && touch.pageY < r.bottom) {

                            if (!allowMultiple && df.querySelector('.draggable-item')) {
                                const old = df.querySelector('.draggable-item');
                                df.removeChild(old);
                                dragContainer.appendChild(old);
                                old.classList.remove('placed');
                            }

                            if (df.childElementCount === 0) df.textContent = '';

                            df.appendChild(draggable);
                            draggable.classList.add('placed');
                        }
                    });
                    updateElementState(dragContainer, dropFields);
                }, { once:true });
            });
        });

        dropFields.forEach(df => {
            df.addEventListener('dragover', e => { e.preventDefault(); df.classList.add('hovered'); });
            df.addEventListener('dragleave', () => df.classList.remove('hovered'));

            df.dataset.placeholder = df.textContent.trim();

            df.addEventListener('drop', e => {
                e.preventDefault();
                df.classList.remove('hovered');
                if (!draggedElement) return;

                if (!allowMultiple && df.querySelector('.draggable-item')) {
                    const old = df.querySelector('.draggable-item');
                    df.removeChild(old);
                    dragContainer.appendChild(old);
                    old.classList.remove('placed');
                }

                if (df.childElementCount === 0) df.textContent = ''; 

                df.appendChild(draggedElement);
                draggedElement.classList.add('placed');
                updateElementState(dragContainer, dropFields);
            });

            df.addEventListener('click', e => {
                const clickedItem = e.target.closest('.draggable-item');
                if (!clickedItem || !df.contains(clickedItem)) return; 

                df.removeChild(clickedItem);
                dragContainer.appendChild(clickedItem);
                clickedItem.classList.remove('placed');

                if (!df.querySelector('.draggable-item')) {
                    df.textContent = df.dataset.placeholder;
                }

                updateElementState(dragContainer, dropFields);
            });
        });
    }

    function updateElementState(dragContainer, dropFields) {
        const allDraggables = dragContainer.querySelectorAll('.draggable-item');

        const placed = new Set();
        dropFields.forEach(df => {
            df.querySelectorAll('.draggable-item').forEach(item => placed.add(item));
        });

        allDraggables.forEach(item => {
            if (placed.has(item)) {
                item.classList.add('used');
                item.classList.remove('unused');
            } else {
                item.classList.add('unused');
                item.classList.remove('used');
            }
        });
    }

    function formatEvaluation(evaluationText) {
        return evaluationText.split('*').map(part => {
            if (part.trim()) {
                return `<p>${part.trim()}</p>`;
            }
        }).join('');
    }

    let selectedDifficulty = 'mittel';

    function selectDifficulty(level) {
        selectedDifficulty = level;

        document.querySelectorAll('.difficulty-button').forEach(button => {
            button.classList.remove('selected');
        });

        document.getElementById(level).classList.add('selected');
    }

    let answeredQuestions = 0; 
    let totalQuestions = 0;
    document.getElementById('recommendation-container').style.display = 'none';

    function trackProgress(isCorrect) {
        answeredQuestions++; 

        fetch('/track-progress', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ correct: isCorrect })
        })
        .then(response => response.json())
        .then(data => {
            console.log("Fortschritt getrackt:", data.status);
        })
        .catch(error => console.error('Fehler beim Tracking des Fortschritts:', error));

        console.log(`Answered: ${answeredQuestions}, Total: ${totalQuestions}`);
        if (answeredQuestions === totalQuestions) {
            console.log("Calling getRecommendation()");
            getRecommendation(); 
        }
    }

    document.addEventListener('DOMContentLoaded', function() {
        const generateButton = document.getElementById('generate-questions');
        const quizContainer = document.getElementById('quiz-container');
        const loadingIndicator = document.getElementById('loading');
        const progressBar = document.getElementById('progress');
        const progressText = document.getElementById('progress-text');
        const messageContainer = document.createElement('div');
        const pdfUpload = document.getElementById('pdf-upload');
        const uploadPdfBtn = document.getElementById('upload-pdf-btn');
        const pdfSelect = document.getElementById('pdf-select');

        messageContainer.id = 'system-message';
        messageContainer.className = 'hidden'; 
        generateButton.parentElement.appendChild(messageContainer);

        function setError(message) {
            messageContainer.textContent = message; 
            messageContainer.className = 'error-message';
            const themaInput = document.getElementById('thema-input'); 
            themaInput.value = '';
        }

        uploadPdfBtn.addEventListener('click', function() {
            const file = pdfUpload.files[0];
            if (!file) {
                messageContainer.textContent = "Bitte wählen Sie eine PDF-Datei aus.";
                messageContainer.className = "error-message";
                return;
            }
            const formData = new FormData();
            formData.append('file', file);
            fetch('/upload-pdf', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    messageContainer.textContent = data.message;
                    messageContainer.className = "success-message";
                    fetchPDFList();
                    pdfUpload.value = "";
                    setTimeout(() => {
                        messageContainer.textContent = "";
                        messageContainer.className   = "hidden";
                    }, 5000);
                } else {
                    messageContainer.textContent = data.error || "Ein Fehler ist aufgetreten.";
                    messageContainer.className = "error-message";
                }
            })
            .catch(error => {
                console.error('Fehler beim Hochladen:', error);
                messageContainer.textContent = "Fehler beim Hochladen der PDF.";
                messageContainer.className = "error-message";
            });
        });

        document.getElementById('upload-json-btn').addEventListener('click', function() {
        const msg = document.getElementById('system-message'); 
        msg.textContent = '';
        msg.className   = 'hidden';

        const fileInput = document.getElementById('json-upload');
        const file      = fileInput.files[0];
        if (!file) {
            msg.textContent = 'Bitte wähle eine JSON-Datei aus.';
            msg.className   = 'error-message';
            return;
        }
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
            const data = JSON.parse(e.target.result);

            ['mc_questions','rf_questions','offene_questions','reihenfolge_questions','lueckentext_questions','zuordnungsaufgaben']
            .forEach(k => { if (!Array.isArray(data[k])) data[k] = []; });

            renderQuiz(data);
            document.getElementById('generate-questions').disabled = true;
            document.getElementById('recommendation-container').style.display = 'none';
            } catch (err) {
            console.error(err);
            msg.textContent = 'Ungültiges JSON-Format.';
            msg.className   = 'error-message';
            }
        };
        reader.readAsText(file, 'UTF-8');
        });

        function fetchPDFList() {
            fetch('/list-pdfs')
            .then(response => response.json())
            .then(data => {
                pdfSelect.innerHTML = '<option value="">Keine Quelle ausgewählt</option>';
                data.pdf_files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file;
                    option.textContent = file.replace('.txt', '');
                    pdfSelect.appendChild(option);
                });
            })
            .catch(error => console.error('Fehler beim Abrufen der PDF-Liste:', error));
        }
        fetchPDFList();

        function clearError() {
            messageContainer.textContent = ""; 
            messageContainer.className = 'hidden'; 
        }

        let isGenerating = false;

        generateButton.addEventListener('click', function() {
            const selectedPdf = pdfSelect.value;
            
            if (!selectedPdf) {
                messageContainer.textContent = "Bitte wählen Sie eine Quelle aus.";
                messageContainer.className = "error-message";
                return;
            }

            if (isGenerating) {
                if (messageContainer) {
                    messageContainer.textContent = "Fragen werden bereits generiert. Bitte warten Sie.";
                    messageContainer.className = "error-message";
                }
                return;
            }

            if (messageContainer) {
                messageContainer.textContent = "";
                messageContainer.className = "hidden";
            }

            answeredQuestions = 0;
            totalQuestions = 0;

            isGenerating = true; 
            generateButton.disabled = true; 
            generateButton.textContent = "Generierung läuft..."; 
            generateButton.classList.add('active');

            loadingIndicator.style.display = "block";
            quizContainer.innerHTML = "";
            progressBar.style.width = "0%";
            progressText.innerText = "0%";

            let progress = 0;
            const progressInterval = setInterval(() => {
                if (progress < 95) {
                    progress += 2;
                    progressBar.style.width = progress + "%";
                    progressText.innerText = progress + "%";
                }
            }, 500);

            const apiKey = document.getElementById('openai-key').value.trim();
            fetch('/generate-questions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    api_key: apiKey || null,
                    schwierigkeitsgrad: selectedDifficulty,
                    pdf_file: selectedPdf
                })
            })

            .then(response => response.json())
            .then(data => {
                clearInterval(progressInterval);
                progressBar.style.width = "100%";
                progressText.innerText = "100%";
                setTimeout(() => loadingIndicator.style.display = "none", 500);

                generateButton.disabled = false;
                generateButton.textContent = "Fragen generieren";
                generateButton.classList.remove('active'); 
                isGenerating = false;

                if (data.error) {
                    messageContainer.textContent = data.error;
                } else {
                    renderQuiz(data);
                    messageContainer.textContent = ""; 
                }
            })
            .catch(error => {
                clearInterval(progressInterval);
                loadingIndicator.style.display = "none";
                console.error('Fehler beim Abrufen der Fragen:', error);

                generateButton.disabled = false;
                generateButton.textContent = "Fragen generieren";
                generateButton.classList.remove('active'); 
                isGenerating = false;

                setError("Es konnten keine Fragen für das angegebene Thema generiert werden.");
            });
        });

        function renderQuiz(data) {
            const quizContainer = document.getElementById('quiz-container');
            quizContainer.innerHTML = '';

            const masterList = document.createElement('ol');
            masterList.id = 'quiz-list';
            quizContainer.appendChild(masterList);

            function addSectionHeader(title) {
                const headerLi = document.createElement('li');
                headerLi.classList.add('section-header');
                const h2 = document.createElement('h2');
                h2.textContent = title;
                headerLi.appendChild(h2);
                masterList.appendChild(headerLi);
            }

            if (!Array.isArray(data.lueckentext_questions)) {
                data.lueckentext_questions = [data.lueckentext_questions];
            }

            function countIfNotEmpty(arr) {
            return Array.isArray(arr) && arr.length > 0 ? arr.length : 0;
            }

            totalQuestions = 
            countIfNotEmpty(data.mc_questions) +
            countIfNotEmpty(data.rf_questions) +
            countIfNotEmpty(data.offene_questions) +
            countIfNotEmpty(data.reihenfolge_questions) +
            countIfNotEmpty(data.lueckentext_questions) +
            countIfNotEmpty(data.zuordnungsaufgaben);

            ///////////////////////////////////////////////////////////////////////////
            // 1) Multiple Choice
            ///////////////////////////////////////////////////////////////////////////
            if (data.mc_questions && data.mc_questions.length > 0) {
                addSectionHeader('Multiple Choice Fragen');
                data.mc_questions.forEach(q => {
                    const li = document.createElement('li');
                    li.classList.add('quiz');

                    const p = document.createElement('p');
                    p.classList.add('question');
                    p.innerHTML = q.question;
                    li.appendChild(p);

                    const answersContainer = document.createElement('div');
                    answersContainer.classList.add('mc-answers');
                    q.answers.forEach(ans => {
                        const ansP = document.createElement('p');
                        ansP.innerText = ans;
                        answersContainer.appendChild(ansP);
                    });
                    li.appendChild(answersContainer);

                    const btnContainer = document.createElement('div');
                    btnContainer.classList.add('button-container');
                    const feedback = document.createElement('p');
                    feedback.classList.add('answer');
                    feedback.style.display = 'none';
                    let answered = false;

                    ['A','B','C','D'].forEach(opt => {
                        const btn = document.createElement('button');
                        btn.classList.add('answer-button');
                        btn.innerHTML = opt;
                        btn.addEventListener('click', () => {
                            if (answered) return;
                            answered = true;
                            btn.classList.add('selected');
                            feedback.style.display = 'block';
                            const ok = opt === q.correct_answer;
                            trackProgress(ok);
                            if (ok) {
                                feedback.innerHTML = `<strong>Korrekt.</strong> ${opt} ist richtig.`;
                                feedback.classList.add('correct');
                            } else {
                                feedback.innerHTML = `<strong>Leider falsch.</strong> Richtig: ${q.correct_answer}`;
                                feedback.classList.add('incorrect');
                            }
                            btnContainer.querySelectorAll('button').forEach(b => b.disabled = true);
                        });
                        btnContainer.appendChild(btn);
                    });

                    li.appendChild(btnContainer);
                    li.appendChild(feedback);
                    masterList.appendChild(li);
                });
            }

            ///////////////////////////////////////////////////////////////////////////
            // 2) Richtig/Falsch
            ///////////////////////////////////////////////////////////////////////////
            if (data.rf_questions && data.rf_questions.length > 0) {  
                addSectionHeader('Richtig/Falsch Fragen');
                data.rf_questions.forEach(q => {
                    const li = document.createElement('li');
                    li.classList.add('quiz');

                    const p = document.createElement('p');
                    p.classList.add('question');
                    p.innerHTML = q.question;
                    li.appendChild(p);

                    const tfContainer = document.createElement('div');
                    tfContainer.classList.add('true-false-answers');
                    const feedback = document.createElement('p');
                    feedback.classList.add('feedback-true-false');
                    feedback.style.display = 'none';

                    const trueBtn = document.createElement('button');
                    trueBtn.classList.add('true-false-btn','true-btn');
                    trueBtn.innerHTML = 'Richtig';
                    const falseBtn = document.createElement('button');
                    falseBtn.classList.add('true-false-btn','false-btn');
                    falseBtn.innerHTML = 'Falsch';

                    const disableTF = () => { trueBtn.disabled = falseBtn.disabled = true; };

                    trueBtn.addEventListener('click', () => {
                        const ok = q.correct_answer === 'Richtig';
                        feedback.style.display = 'block';
                        feedback.innerHTML = ok ? 'Korrekt.' : 'Leider falsch.';
                        feedback.classList.toggle('correct', ok);
                        feedback.classList.toggle('incorrect', !ok);
                        trueBtn.classList.add('selected');
                        disableTF();
                        trackProgress(ok);
                    });
                    falseBtn.addEventListener('click', () => {
                        const ok = q.correct_answer === 'Falsch';
                        feedback.style.display = 'block';
                        feedback.innerHTML = ok ? 'Korrekt.' : 'Leider falsch.';
                        feedback.classList.toggle('correct', ok);
                        feedback.classList.toggle('incorrect', !ok);
                        falseBtn.classList.add('selected');
                        disableTF();
                        trackProgress(ok);
                    });

                    tfContainer.appendChild(trueBtn);
                    tfContainer.appendChild(falseBtn);
                    li.appendChild(tfContainer);
                    li.appendChild(feedback);
                    masterList.appendChild(li);
                });
            }

            ///////////////////////////////////////////////////////////////////////////
            // 3) Offene Fragen
            ///////////////////////////////////////////////////////////////////////////
            if (data.offene_questions && data.offene_questions.length > 0) {  
                addSectionHeader('Offene Fragen');
                data.offene_questions.forEach(q => {
                    const li = document.createElement('li');
                    li.classList.add('quiz');

                    const p = document.createElement('p');
                    p.classList.add('question');
                    p.innerHTML = q.question;
                    li.appendChild(p);

                    const textarea = document.createElement('textarea');
                    textarea.classList.add('answer-input');
                    textarea.placeholder = 'Antwort eingeben';
                    li.appendChild(textarea);

                    const btn = document.createElement('button');
                    btn.classList.add('evaluate-answer');
                    btn.innerHTML = 'Antwort bewerten';

                    const spinnerContainer = document.createElement('div');
                    spinnerContainer.classList.add('spinner-container');
                    spinnerContainer.style.display = 'none';

                    const spinner = document.createElement('div');
                    spinner.classList.add('spinner');

                    const loadingText = document.createElement('span');
                    loadingText.innerText = 'Analyse läuft...';
                    spinnerContainer.appendChild(spinner);
                    spinnerContainer.appendChild(loadingText);

                    const feedback = document.createElement('p');
                    feedback.classList.add('feedback-open');

                    li.appendChild(btn);
                    li.appendChild(spinnerContainer);
                    li.appendChild(feedback);

                    feedback.style.display = 'none';

                    masterList.appendChild(li);

                    btn.addEventListener('click', async () => {
                        if (q.evaluated) {
                            feedback.textContent = 'Bereits bewertet.';
                            feedback.style.display = 'block';
                            feedback.classList.replace('feedback-open','error-message');
                            return;
                        }
                        const ans = textarea.value.trim();
                        if (!ans) {
                            feedback.textContent = 'Bitte eine Antwort eingeben.';
                            feedback.classList.remove('feedback-open');
                            feedback.classList.add('error-message');
                            feedback.style.display = 'block';
                            textarea.focus();
                            return;
                        }
                        q.evaluated = true;

                        spinnerContainer.style.display = 'flex';
                        await new Promise(res => setTimeout(res, 300));

                        try {
                            const apiKey = document.getElementById('openai-key').value.trim();
                            const res = await fetch('/evaluate-answer', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    api_key:     apiKey || null,
                                    question_id: q.id,
                                    question:    q.question,
                                    answer:      ans,
                                    source:      q.source 
                                })
                            });
                            const data = await res.json();

                            spinnerContainer.style.display = 'none';

                            feedback.style.display = 'block';
                            feedback.innerHTML = data.evaluation;
                            trackProgress(data.status === 'beantwortet');
                            btn.disabled = true;
                        } catch (err) {
                            spinnerContainer.style.display = 'none';
                            feedback.style.display = 'block';
                            feedback.innerHTML = 'Fehler bei der Analyse.';
                        }
                    });
                });
            }

            ///////////////////////////////////////////////////////////////////////////
            // 4) Reihenfolge
            ///////////////////////////////////////////////////////////////////////////
            if (data.reihenfolge_questions && data.reihenfolge_questions.length > 0) {      
                addSectionHeader('Reihenfolgenaufgaben');
                data.reihenfolge_questions.forEach(q => {
                    const li = document.createElement('li');
                    li.classList.add('quiz');

                    const p = document.createElement('p');
                    p.classList.add('question');
                    p.innerHTML = q.description;
                    li.appendChild(p);

                    const fields = document.createElement('div');
                    fields.classList.add('field-container');
                    q.correct_order.forEach((_, i) => {
                        const df = document.createElement('div');
                        df.classList.add('drop-field');
                        df.dataset.index = i+1;
                        df.textContent = i+1;
                        fields.appendChild(df);
                    });
                    li.appendChild(fields);

                    const dragBox = document.createElement('div');
                    dragBox.classList.add('drag-container');
                    q.elements.forEach(el => {
                        const item = document.createElement('div');
                        item.classList.add('draggable-item');
                        item.draggable = true;
                        item.innerText = el;
                        item.addEventListener('dragstart', () => draggedElement = item);
                        item.addEventListener('dragend', () => draggedElement = null);
                        dragBox.appendChild(item);
                    });
                    li.appendChild(dragBox);

                    const dropZones = fields.querySelectorAll('.drop-field');
                    dropZones.forEach(df => {
                        df.dataset.placeholder = df.textContent;
                        df.addEventListener('dragover', e => { e.preventDefault(); df.classList.add('hovered'); });
                        df.addEventListener('dragleave', () => df.classList.remove('hovered'));
                        df.addEventListener('drop', e => {
                            e.preventDefault();
                            df.classList.remove('hovered');
                            if (draggedElement) {
                                if (df.firstChild) {
                                    if (df.firstChild.nodeType === Node.ELEMENT_NODE) {  
                                        const old = df.removeChild(df.firstChild);
                                        dragBox.appendChild(old);
                                        old.classList.remove('placed');
                                    } else {
                                        df.removeChild(df.firstChild);
                                    }
                                }
                                df.textContent = '';
                                df.appendChild(draggedElement);
                                updateElementState(dragBox, dropZones);
                            }
                        });
                        df.addEventListener('click', () => {
                        const child = df.querySelector('.draggable-item');
                        if (!child) return;  
                        df.removeChild(child);          
                        dragBox.appendChild(child);
                        child.classList.remove('placed');
                        df.textContent = df.dataset.placeholder;
                        updateElementState(dragBox, dropZones);
                        });
                    });

                    const btn = document.createElement('button');
                    btn.classList.add('evaluate-answer');
                    btn.innerText = 'Antwort überprüfen';
                    const feedback = document.createElement('p');
                    feedback.classList.add('feedback-open');
                    feedback.style.display = 'none';

                    btn.addEventListener('click', () => {
                        feedback.style.display = 'block';
                        const userOrder = Array.from(fields.children).map(f => f.firstChild ? f.firstChild.textContent : null);
                        const ok = userOrder.every((v,i) => v === q.correct_order[i]);
                        feedback.innerHTML = ok
                            ? '<strong>Korrekt.</strong> Reihenfolge stimmt.'
                            : `<strong>Leider falsch.</strong> Richtig: ${q.correct_order.join(', ')}`;
                        feedback.classList.toggle('correct', ok);
                        feedback.classList.toggle('incorrect', !ok);
                        btn.disabled = true;
                        dropZones.forEach(z => z.style.pointerEvents = 'none');
                        trackProgress(ok);
                    });

                    li.appendChild(btn);
                    li.appendChild(feedback);
                    masterList.appendChild(li);
                });
            }

            ///////////////////////////////////////////////////////////////////////////
            // 5) Lückentext
            ///////////////////////////////////////////////////////////////////////////
            if (data.lueckentext_questions && data.lueckentext_questions.length > 0) {   
                addSectionHeader('Lückentext-Aufgaben');
                data.lueckentext_questions.forEach(q => {
                    const li = document.createElement('li');
                    li.classList.add('quiz','lueckentext-container');

                    const instr = document.createElement('p');
                    instr.classList.add('question'); 
                    instr.classList.add('instruction-text');
                    instr.innerHTML = `Anweisung: ${q.instruction}`;
                    li.appendChild(instr);

                    const txt = document.createElement('p');
                    const parts = q.text.split('___');
                    parts.forEach((part,i) => {
                        txt.appendChild(document.createTextNode(part));
                        if (i < parts.length - 1) {
                            const inp = document.createElement('input');
                            inp.type = 'text';
                            inp.classList.add('lueckentext-input');
                            inp.dataset.index = i;
                            txt.appendChild(inp);
                        }
                    });
                    li.appendChild(txt);

                    const hints = document.createElement('div');
                    hints.classList.add('hint-container');
                    hints.innerHTML = `<strong>Hinweise:</strong> ${q.hints.join(', ')}`;
                    li.appendChild(hints);

                    const btn = document.createElement('button');
                    btn.classList.add('evaluate-answer');
                    btn.innerText = 'Antwort überprüfen';
                    const feedback = document.createElement('p');
                    feedback.classList.add('feedback-open');
                    feedback.style.display = 'none';

                    btn.addEventListener('click', () => {
                        feedback.style.display = 'block';
                        const answers = Array.from(txt.querySelectorAll('input')).map(i => i.value.trim());
                        const ok = answers.every((a,i) => a.toLowerCase() === q.correct_answers[i].toLowerCase());
                        feedback.innerHTML = ok
                            ? '<strong>Korrekt.</strong> Alle richtig.'
                            : `<strong>Leider falsch.</strong> Richtig: ${q.correct_answers.join(', ')}`;
                        feedback.classList.toggle('correct', ok);
                        feedback.classList.toggle('incorrect', !ok);
                        trackProgress(ok);
                        btn.disabled = true;
                    });

                    li.appendChild(btn);
                    li.appendChild(feedback);
                    masterList.appendChild(li);
                });
            }

            ///////////////////////////////////////////////////////////////////////////
            // 6) Zuordnung
            ///////////////////////////////////////////////////////////////////////////
            if (data.zuordnungsaufgaben && data.zuordnungsaufgaben.length > 0) { 
                addSectionHeader('Zuordnungsaufgaben');
                data.zuordnungsaufgaben.forEach(q => {
                    const li = document.createElement('li');
                    li.classList.add('quiz');

                    const p = document.createElement('p');
                    p.classList.add('question');
                    p.innerText = q.description;
                    li.appendChild(p);

                    const fields = document.createElement('div');
                    fields.classList.add('field-container');
                    q.categories.forEach(cat => {
                    const title = document.createElement('div');
                    title.classList.add('category-title');
                    title.innerText = cat;
                    fields.appendChild(title);

                    const zone = document.createElement('div');
                    zone.classList.add('drop-field');
                    zone.dataset.category = cat;
                    fields.appendChild(zone);
                    });
                    li.appendChild(fields);

                    const dragBox = document.createElement('div');
                    dragBox.classList.add('drag-container');
                    q.elements.forEach(el => {
                    const item = document.createElement('div');
                    item.classList.add('draggable-item');
                    item.draggable = true;
                    item.innerText = el;
                    item.addEventListener('dragstart', () => draggedElement = item);
                    item.addEventListener('dragend',   () => draggedElement = null);
                    dragBox.appendChild(item);
                    });
                    li.appendChild(dragBox);

                    const dropZones = fields.querySelectorAll('.drop-field');
                    setupDragAndDrop(dragBox, dropZones, true);

                    const btn = document.createElement('button');
                    btn.classList.add('evaluate-answer');
                    btn.innerText = 'Antwort überprüfen';
                    const feedback = document.createElement('p');
                    feedback.classList.add('feedback-open');
                    feedback.style.display = 'none';

                    btn.addEventListener('click', () => {
                    feedback.style.display = 'block';
                    let ok = true;
                    dropZones.forEach(z => {
                        const placed   = Array.from(z.children).map(c => c.innerText);
                        const expected = q.correctMappings[z.dataset.category] || [];
                        if (placed.length !== expected.length ||
                            !placed.every(e => expected.includes(e))
                        ) ok = false;
                    });

                    if (ok) {
                        feedback.innerHTML = '<strong>Korrekt.</strong> Zuordnung stimmt.';
                        feedback.classList.add('correct');
                    } else {
                        let html = '<strong>Leider falsch. Korrekte Zuordnungen:</strong><ul style="list-style-type: disc; padding-left:1.2em;">';
                        for (const [cat, elems] of Object.entries(q.correctMappings)) {
                            html += `<li><strong>${cat}:</strong> ${elems.join(', ')}</li>`;
                        }
                        html += '</ul>';
                        feedback.innerHTML = html;
                        feedback.classList.add('incorrect');
                    }

                    btn.disabled = true;
                    dropZones.forEach(z => z.style.pointerEvents = 'none');
                    trackProgress(ok);
                    });

                    li.appendChild(btn);
                    li.appendChild(feedback);
                    masterList.appendChild(li);
                });
            }
        }

        const resetBtn = document.getElementById('reset-btn');
        resetBtn.addEventListener('click', () => {
            document.getElementById('pdf-select').value         = "";
            document.getElementById('json-upload').value        = "";
            document.getElementById('generate-questions').disabled = false;
            document.getElementById('quiz-container').innerHTML = "";
            document.getElementById('recommendation-container').style.display = 'none';
            resetBtn.classList.add('hidden');
            const msg = document.getElementById('system-message');
            msg.textContent = "";
            msg.className   = "hidden";
            answeredQuestions = 0;
            totalQuestions    = 0;
        });
    });    

    function getRecommendation() {
    fetch('/evaluate-level')
        .then(res => res.json())
        .then(data => {
        const recCont = document.getElementById('recommendation-container');
        const recText = document.getElementById('recommendation-text');

        recText.innerHTML = data.recommendation;
        recText.className = 'recommendation-keep';

        recCont.style.display = 'block';
        document.getElementById('reset-btn').classList.remove('hidden');
        })
        .catch(console.error);
    }

    """
    with open(STATIC_DIR / "script.js",  "w", encoding="utf-8") as f:
        f.write(js_content)

@app.route("/")
def index():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    if not getattr(sys, "frozen", False):
        generiere_html()
        generiere_css()
        generiere_js()

    app.run(host="0.0.0.0", port=5000)

