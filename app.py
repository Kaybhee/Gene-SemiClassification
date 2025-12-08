from pathlib import Path
import re
import logging
from typing import Optional

from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
# For rendering static files, such as html and css files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
app = FastAPI()





try:
    import rag
except Exception:
    rag = None

import joblib

# Allow CORS from local frontends. Use a restrictive list in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://127.0.0.1:5500", "http://localhost:5500", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger("gene_api")

app.mount("/static", StaticFiles(directory="static"), name="static")


def safe_load(path: Path):
    
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return joblib.load(path)


def get_resource_dir():
    return Path(__file__).resolve().parent


def simple_preprocess(text: str) -> str:
    # A lightweight fallback preprocess in case NLTK is not available
    alpha_or_numeric = "[^a-zA-Z0-9- ]"
    txt = re.sub(alpha_or_numeric, " ", text)
    txt = txt.lower()
    # minimal stopwords
    stopwords_min = {"the","and","or","of","in","on","with","a","an","to","for","by","at"}
    tokens = [t for t in txt.split() if t not in stopwords_min and len(t) > 1]
    return " ".join(tokens)


@app.on_event("startup")
def load_models():
    global model, vect, pca
    resdir = get_resource_dir()/ "gene_pipeline"
    try:
        model = safe_load(resdir / "gene_signature_model.pkl")
        vect = safe_load(resdir / "tfidf_vectorizer.pkl")
        pca = safe_load(resdir / "pca_transformer.pkl")
        logger.info("Model and pipeline loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model artifacts: %s", e)
        # re-raise so startup fails visibly when using uvicorn
        raise


def preprocess_text(text: str) -> str:
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem.wordnet import WordNetLemmatizer

        stop_words = set(stopwords.words('english'))
        wordnet_lemm = WordNetLemmatizer()
        alpha_or_numeric = "[^a-zA-Z0-9- ]"
        pre_txt = re.sub(alpha_or_numeric, " ", text)
        pre_txt = pre_txt.lower()
        sample_words = [wordnet_lemm.lemmatize(w) for w in pre_txt.split() if w not in stop_words and len(w) > 1]
        return ' '.join(sample_words)
    except Exception:
        # If nltk not installed or resources missing, use simple fallback
        return simple_preprocess(text)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Completed request: {request.method} {request.url} -> {response.status_code}")
        return response
    except Exception as e:
        logger.exception("Unhandled error processing request: %s", e)
        raise


# @app.get("/")
# def root():
#     return {"status": "ok", "message": "Gene signature API. POST to /classify with form field 'sample'"}
# Serve frontend index.html
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index_path = Path("static/index.html")
    if index_path.exists():
        return index_path.read_text()
    return "<h2>index.html not found inside /static/</h2>"

@app.post("/classify")
async def classify_sample(sample: str = Form(...)):
    try:
        cleaned = preprocess_text(sample)
        X_vec = vect.transform([cleaned])
        X_pca = pca.transform(X_vec.toarray())
        pred = model.predict(X_pca)[0]
        # Map numeric prediction to human label if binary (common in this notebook)
        label_map = {1: 'ctrl', 0: 'pert'}
        mapped = label_map.get(int(pred), str(pred))
        conf: Optional[float] = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_pca)
            # binary case: take probability of positive class if two columns
            if proba.ndim == 2 and proba.shape[1] == 2:
                conf = float(proba[0, 1])
            else:
                conf = float(proba.max())

        # Extract genes from input text and retrieve RAG insights
        extracted_genes = rag.extract_genes_from_text(sample) if rag else []
        rag_insights = None
        if rag:
            try:
                rag_insights = rag.retrieve_insights(mapped, extracted_genes)
            except Exception as e:
                logger.warning("RAG retrieval failed: %s", e)

        logger.info("Classify request processed. sample_len=%d pred=%s conf=%s genes=%s", len(sample), str(pred), str(conf), extracted_genes)
        result = {
            "predicted_signature_raw": str(pred),
            "predicted_signature": mapped,
            "confidence": conf,
            "extracted_genes": extracted_genes
        }
        if rag_insights:
            result["rag_insights"] = rag_insights
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Error during classification: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post('/predict')
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail='Only CSV uploads supported')
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Could not parse CSV: {e}')

    # Determine text features: prefer a 'feature' column, otherwise concatenate object columns
    if 'feature' in df.columns:
        docs = df['feature'].astype(str)
    else:
        obj_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype == 'string']
        if obj_cols:
            docs = df[obj_cols].astype(str).agg(' '.join, axis=1)
        else:
            raise HTTPException(status_code=400, detail='CSV must contain a `feature` column or one/more text/object columns to build text features. Numeric-expression CSVs are not supported by this endpoint.')

    # Preprocess text, vectorize, project and predict using loaded artifacts (vect, pca, model)
    try:
        cleaned = docs.apply(lambda t: preprocess_text(str(t)))
        X_tfidf = vect.transform(cleaned.tolist())
        X_array = X_tfidf.toarray()
        X_proj = pca.transform(X_array)

        preds = model.predict(X_proj)
        probs = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_proj)
            probs = proba.tolist()
    except Exception as e:
        logger.exception('Error during CSV text prediction pipeline: %s', e)
        raise HTTPException(status_code=500, detail=f'Error during prediction pipeline: {e}')

    results = []
    label_map = {1: 'ctrl', 0: 'pert'}
    for i, p in enumerate(preds):
        raw = p
        mapped = label_map.get(int(p), str(p)) if isinstance(p, (int, float)) or (isinstance(p, str) and p.isdigit()) else str(p)
        r = {'row': i, 'prediction_raw': str(raw), 'prediction': mapped}
        if probs is not None:
            r['probability'] = probs[i]
        
        # Extract genes from row text and attach RAG insights
        row_text = docs.iloc[i] if hasattr(docs, 'iloc') else str(docs[i])
        extracted_genes = rag.extract_genes_from_text(row_text) if rag else []
        r['extracted_genes'] = extracted_genes
        
        if rag is not None:
            try:
                rag_insights = rag.retrieve_insights(mapped, extracted_genes)
                r['rag_insights'] = rag_insights
            except Exception as e:
                logger.warning("RAG retrieval for row %d failed: %s", i, e)
        
        results.append(r)

    return {'n_rows': len(df), 'results': results}


if __name__ == "__main__":
    import uvicorn
    # uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)