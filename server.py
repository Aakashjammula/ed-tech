from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import EdTechAssistant
from typing import Dict, List
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import tempfile
import shutil

load_dotenv()
app = FastAPI()
assistant = EdTechAssistant()
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("./index.html")

@app.get("/textbooks")
async def serve_textbooks():
    return FileResponse("./textbooks.html")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post("/query")
async def process_query(query: Query) -> Dict[str, str]:
    try:
        # Get the answer directly from the ask_question method
        _, answer = assistant.ask_question(query.query)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/upload-textbooks")
async def upload_textbooks(files: List[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            if not file.filename.endswith('.pdf'):
                results.append({"file": file.filename, "status": "failed", "message": "Only PDF files are supported"})
                continue
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                # Copy the uploaded file content to the temporary file
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name
            
            # Process the PDF file
            try:
                result = assistant.process_pdfs([type('obj', (object,), {'name': temp_file_path})])
                results.append({
                    "file": file.filename,
                    "status": "success",
                    "message": f"Successfully processed {file.filename}"
                })
            except Exception as e:
                results.append({
                    "file": file.filename,
                    "status": "failed",
                    "message": f"Error processing file: {str(e)}"
                })
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading textbooks: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)