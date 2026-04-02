import requests
import json
import sys
import os

# --- CONFIGURATION ---
API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"
CONTEXT_WINDOW = 4000  # Taille approximative des morceaux (en caractères)

# Le Prompt Système : C'est ici que vous définissez la "persona" de l'IA.
SYSTEM_PROMPT = """
Tu es un éditeur scientifique senior en physique. Ton rôle est de relire des extraits d'articles de recherche.
Ton objectif : Améliorer la clarté, la fluidité et l'anglais (ou le français) académique.
CONTRAINTES STRICTES :
1. NE JAMAIS modifier les équations LaTeX (entre $ ou $$) ou dans les rubriques "equation" o.
2. NE JAMAIS inventer de contenu ou de faits.
3. Pour chaque modification suggérée, donne :
   - La phrase originale.
   - La suggestion améliorée.
   - Une justification claire et précise (grammaire, style, ambiguïté).
4. Si le texte est déjà parfait, indique-le simplement.
Format de sortie souhaité : Markdown.
"""

def query_local_llm(chunk_text):
    """Envoie un morceau de texte au modèle local via l'API Ollama."""
    
    # Construction de la requête (Payload)
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{SYSTEM_PROMPT}\n\nVoici le texte à analyser :\n{chunk_text}",
        "stream": False,       # On attend la réponse complète (pas de streaming mot à mot)
        "temperature": 0.2,    # Très important : Basse température = Plus déterministe/Sérieux
        "options": {
            "num_ctx": 8192    # Force la fenêtre de contexte
        }
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() # Vérifie les erreurs HTTP
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        return f"Erreur de connexion au modèle : {e}"

def process_file(filepath):
    """Lit le fichier, le découpe et agrège les critiques."""
    
    if not os.path.exists(filepath):
        print(f"Erreur : Le fichier {filepath} n'existe pas.")
        return

    print(f"--- Analyse de {filepath} en cours avec {MODEL_NAME} ---")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Découpage basique par paragraphes pour ne pas saturer la mémoire du modèle
    # Dans un script de prod, on ferait un découpage plus intelligent (par section LaTeX)
    paragraphs = full_text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < CONTEXT_WINDOW:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk)
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk)

    # Traitement des chunks
    full_report = f"# Rapport de révision pour {filepath}\n\n"
    
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        print(f"Traitement de la section {i+1}/{total_chunks}...")
        critique = query_local_llm(chunk)
        full_report += f"## Section {i+1}\n{critique}\n\n---\n\n"

    # Sauvegarde du rapport
    output_filename = f"RAPPORT_{os.path.basename(filepath)}.md"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    print(f"\nTerminé ! Le rapport a été sauvegardé sous : {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 reviewer.py <votre_article.tex ou .txt>")
    else:
        process_file(sys.argv[1])