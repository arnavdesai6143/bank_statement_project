from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
from ollama import Client
import pandas as pd
import zipfile
import io
import json
import re
import fitz

app = FastAPI()

app.add_middleware(
   CORSMiddleware,
   allow_origins=[
       "http://localhost:3000",
       "http://127.0.0.1:3000",
       "http://localhost:8000",
       "http://127.0.0.1:8000"
   ],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

ollama_client = Client()

class CashFlowResponse(BaseModel):
   filename: str
   total_credit: float
   total_debit: float
   total_cashflow: float

def extract_bank_table_from_pdf(pdf_file: UploadFile) -> pd.DataFrame:
    pdf_bytes = pdf_file.file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    all_text = ""
    for page in doc:
        all_text += page.get_text()

    ollama_prompt = f"""
You are an expert in financial documents. The following text is extracted from a bank statement PDF.
Your task is to locate and extract only the bank transaction table from it.

Extract only lines that represent transactions like dates, transaction descriptions, credits, debits, balances, etc.
Avoid headers, footers, page numbers, or unrelated content.

Here is the text:
{all_text}
"""

    response = ollama_client.chat(model="mistral", messages=[{
        "role": "user",
        "content": ollama_prompt
    }])

    table_text = response["message"]["content"]

    lines = table_text.strip().splitlines()
    clean_lines = [line for line in lines if len(line.split()) >= 3]

    csv_data = []
    for line in clean_lines:
        cells = re.split(r'\s{2,}|\t|,', line.strip())
        csv_data.append(cells)

    df = pd.DataFrame(csv_data)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)

    if df.shape[0] > 1 and not df.iloc[0].isnull().any():
        df.columns = df.iloc[0]
        df = df[1:]

    return df

def parse_bank_statement(file_content: bytes) -> pd.DataFrame:
   try:
       df = pd.read_csv(io.BytesIO(file_content), encoding='latin-1')
   except UnicodeDecodeError:
       try:
           df = pd.read_csv(io.BytesIO(file_content), encoding='cp1252')
       except Exception as e:
           raise ValueError("Unsupported CSV format or encoding.")
   except Exception:
       try:
           df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
       except Exception:
           raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")
   return df

def normalize_dataframe(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    # Extract and coerce metadata
    date_col = str(metadata.get("date_column", "")).strip()
    desc_col = str(metadata.get("description_column", "")).strip()
    amount_col = str(metadata.get("amount_column", "")).strip()
    type_col = metadata.get("type_column")
    credit_label = metadata.get("credit_indicator")
    debit_label = metadata.get("debit_indicator")
    credit_col = str(metadata.get("separate_credit_column", "")).strip()
    debit_col = str(metadata.get("separate_debit_column", "")).strip()

    if isinstance(type_col, int):
        type_col = str(type_col)
    if isinstance(credit_label, int):
        credit_label = str(credit_label)
    if isinstance(debit_label, int):
        debit_label = str(debit_label)
    if type_col is not None:
        type_col = str(type_col).strip()

    # Validate required columns
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in file.")
    if desc_col not in df.columns:
        raise ValueError(f"Description column '{desc_col}' not found in file.")

    standard_df = pd.DataFrame()
    standard_df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    standard_df["Transaction Name"] = df[desc_col].astype(str)

    # Case 1: Separate credit & debit columns
    if credit_col in df.columns and debit_col in df.columns:
        standard_df["Credit"] = pd.to_numeric(df[credit_col], errors="coerce").fillna(0).abs()
        standard_df["Debit"] = pd.to_numeric(df[debit_col], errors="coerce").fillna(0).abs()

    # Case 2: Single amount column + type column
    elif amount_col in df.columns:
        amount_series = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
        if type_col and credit_label and debit_label and type_col in df.columns:
            df_type = df[type_col].astype(str).str.lower()
            credit_label = str(credit_label).lower()
            debit_label = str(debit_label).lower()

            standard_df["Credit"] = amount_series.where(df_type == credit_label, 0).abs()
            standard_df["Debit"] = amount_series.where(df_type == debit_label, 0).abs()
        else:
            # Case 3: Signed values
            standard_df["Credit"] = amount_series.where(amount_series > 0, 0).abs()
            standard_df["Debit"] = amount_series.where(amount_series < 0, 0).abs()
    else:
        raise ValueError("Could not determine credit/debit columns from metadata or fallback.")

    return standard_df



def analyze_cashflow_with_ollama(df: pd.DataFrame) -> dict:
   try:
       # 🧹 Use only the first 40 rows for prompting to avoid overload
       df_string = df.head(40).to_csv(index=False)
       prompt = f"""
You are a financial data analyst helping to process raw bank statements.


The uploaded CSV may contain noisy, unconventional, or unclear column names such as:
- "Withdrawn Amt", "Deposited Amt"
- "Dr", "Cr", "DR", "CR"
- "Txn Amount" + "Type"
- "Amount Changed"
- "Signed Amount"
- "Income", "Expenditure", etc.
Your task is to analyze the structure of the table and return a JSON that clearly identifies the roles of each column, even if the names are not standard.
Here is how the JSON format should look:
{{
 "date_column": "<column containing date>",
 "description_column": "<column with transaction description or merchant>",
 "amount_column": "<column that contains both debit and credit values, signed or unsigned>",
 "type_column": "<column indicating type of transaction like Credit/Debit/CR/DR (if present)>",
 "credit_indicator": "<exact label in the type column indicating credit (e.g., 'Credit', 'CR')>",
 "debit_indicator": "<exact label in the type column indicating debit (e.g., 'Debit', 'DR')>",
 "separate_credit_column": "<column name for credit values if stored separately, else null>",
 "separate_debit_column": "<column name for debit values if stored separately, else null>"
}}
Only return the JSON. Do not include any explanation, text, or comments.
Examples:
- If the file has "Deposited Amt" and "Withdrawn Amt", map them to "separate_credit_column" and "separate_debit_column"
- If the file has a signed amount column like "Amount Changed", use "amount_column" only
- If the file has a "Type" column with values "CR"/"DR", provide that as "type_column"
Now analyze this uploaded bank statement:
{df_string}
"""
       response = ollama_client.chat(model='mistral', messages=[{
           'role': 'user',
           'content': prompt,
       }])
       response_text = response['message']['content'].strip()
       try:
           metadata = json.loads(response_text)
       except json.JSONDecodeError:
           json_match = re.search(r'{[\s\S]*?}', response_text)
           if not json_match:
               raise ValueError("Could not extract JSON metadata from Ollama response.")
           metadata = json.loads(json_match.group(0))
       for key, value in metadata.items():
           if isinstance(value, list):
               metadata[key] = value[0]

       standard_df = normalize_dataframe(df, metadata)

       total_credit = standard_df["Credit"].abs().sum()
       total_debit = standard_df["Debit"].abs().sum()
       total_cashflow = total_credit - total_debit

       return {
           "total_credit": round(total_credit, 2),
           "total_debit": round(total_debit, 2),
           "total_cashflow": round(total_cashflow, 2)
       }

   except Exception as e:
       print(f"Error during cashflow analysis: {e}")
       return {
           "total_credit": 0.0,
           "total_debit": 0.0,
           "total_cashflow": 0.0
       }

@app.post("/analyze_statements/", response_model=List[CashFlowResponse])
async def analyze_statements(files: List[UploadFile] = File(...)):
   results = []
   try:
       file = files[0]
       filename = file.filename

       if filename.endswith('.zip'):
           with zipfile.ZipFile(io.BytesIO(await file.read())) as zip_file:
               for file_name in zip_file.namelist():
                   if not file_name.endswith(('.csv', '.xlsx')) or file_name.startswith("__MACOSX") or ".DS_Store" in file_name:
                       continue
                   with zip_file.open(file_name) as inner_file:
                       file_content = inner_file.read()
                       try:
                           df = parse_bank_statement(file_content)
                           analysis = analyze_cashflow_with_ollama(df)
                           results.append(CashFlowResponse(
                               filename=file_name,
                               **analysis
                           ))
                       except Exception as e:
                           print(f"Error processing file {file_name}: {e}")
                           continue
       else:
           df = parse_bank_statement(await file.read())
           analysis = analyze_cashflow_with_ollama(df)
           results.append(CashFlowResponse(
               filename=filename,
               **analysis
           ))

       return results
   except zipfile.BadZipFile:
       raise HTTPException(status_code=400, detail="Invalid zip file.")
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/analyze_pdf_statement/", response_model=CashFlowResponse)
async def analyze_pdf_statement(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        df = extract_bank_table_from_pdf(file)
        analysis = analyze_cashflow_with_ollama(df)
        return CashFlowResponse(filename=file.filename, **analysis)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the PDF: {e}")

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)
