import pandas as pd
import requests
import json

# Configuration
EXCEL_FILE = "questions.xlsx"  # Path to your Excel file
SHEET_NAME = "Sheet1"          # Excel sheet name
QUESTION_COLUMN = "Question"    # Column with questions
RESPONSE_COLUMN = "AI_Response" # New column for AI responses
OLLAMA_URL = "http://192.168.68.123:11434/api/chat"  # Ollama API endpoint
MODEL_NAME = "Nudge DeepSeek Expert"        # Your model name

def get_ai_response(question):
    """Send a question to Ollama's chat API and return the response."""
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": question}],
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

def main():
    # Read Excel file
    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Ensure the response column exists and is of string type
    if RESPONSE_COLUMN not in df.columns:
        df[RESPONSE_COLUMN] = pd.NA
    df[RESPONSE_COLUMN] = df[RESPONSE_COLUMN].astype("string")  # Explicitly set to string type

    # Process each row
    for index, row in df.iterrows():
        question = row[QUESTION_COLUMN]
        if pd.isna(question) or not str(question).strip():
            df.at[index, RESPONSE_COLUMN] = "Skipped: Empty question"
            continue

        print(f"Processing question: {question}")
        response = get_ai_response(question)
        df.at[index, RESPONSE_COLUMN] = response

    # Save updated Excel file
    try:
        df.to_excel(EXCEL_FILE, sheet_name=SHEET_NAME, index=False)
        print(f"Responses saved to {EXCEL_FILE}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

if __name__ == "__main__":
    # Test Ollama server availability
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        if response.status_code == 200:
            print("Ollama server is running.")
            main()
        else:
            print(f"Ollama server error: Status code {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Error: Ollama server not running. Start it with 'ollama serve'.")