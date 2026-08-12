from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import json, traceback, threading

BASE = Path(__file__).resolve().parent
DATA = json.loads((BASE / "data.json").read_text(encoding="utf-8"))
QUESTIONS = json.loads((BASE / "questions.json").read_text(encoding="utf-8"))

app = FastAPI(title="SQL Interview Practice - PySpark Backend", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

class RunRequest(BaseModel):
    question_id: str = Field(min_length=1)
    code: str = Field(min_length=1, max_length=50000)

spark_lock = threading.Lock()
spark = None
frames = {}

def get_spark():
    global spark, frames
    if spark is None:
        spark = (SparkSession.builder
                 .master("local[*]")
                 .appName("SQLInterviewPractice")
                 .config("spark.ui.enabled", "false")
                 .config("spark.sql.shuffle.partitions", "4")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        for name in DATA["tables"]:
            meta = DATA["data"][name]
            rows = meta["rows"]
            cols = meta["columns"]
            frames[name] = spark.createDataFrame(rows, cols)
    return spark

def clean_value(v):
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v

@app.get("/api/health")
def health():
    return {"status":"ok", "spark":"ready" if spark is not None else "not_started", "tables":len(DATA["tables"]), "questions":len(QUESTIONS)}

@app.get("/api/question/{question_id}")
def question(question_id: str):
    q = next((x for x in QUESTIONS if x["id"] == question_id), None)
    if not q:
        raise HTTPException(404, "Question not found")
    return {"id":q["id"], "title":q["title"], "difficulty":q["difficulty"], "tables":q["tables"], "expected":q["expected"], "columns":q["columns"]}

@app.post("/api/pyspark/run")
def run_pyspark(req: RunRequest):
    q = next((x for x in QUESTIONS if x["id"] == req.question_id), None)
    if not q:
        raise HTTPException(404, "Question not found")
    # IMPORTANT: this endpoint is intended for local/private development.
    # Do not expose arbitrary PySpark execution to the public internet without a real sandbox.
    blocked = ["os.system", "subprocess", "socket", "shutil.rmtree", "__import__", "eval(", "exec(", "open(", "input("]
    low = req.code.lower().replace(" ", "")
    hits = [x for x in blocked if x.lower().replace(" ", "") in low]
    if hits:
        raise HTTPException(400, "Blocked operation in practice code: " + ", ".join(hits))

    with spark_lock:
        try:
            get_spark()
            env = {"spark": spark, "F": F, **frames}
            exec(req.code, {"__builtins__": __builtins__}, env)
            result = env.get("result")
            if result is None or not hasattr(result, "collect"):
                return {"ok":False, "error":"Your code must create the final DataFrame in a variable named result."}
            rows = result.collect()
            cols = result.columns
            payload = [[clean_value(r[c]) for c in cols] for r in rows]
            expected = q.get("expected", [])
            matched = payload == expected
            return {"ok":True, "matched":matched, "columns":cols, "rows":payload[:500], "row_count":len(payload), "expected_columns":q.get("columns",[]), "expected_rows":expected, "message":"Correct answer" if matched else "Code executed successfully, but the result does not match the expected output."}
        except Exception as e:
            return {"ok":False, "error":str(e), "traceback":traceback.format_exc(limit=8)}

@app.post("/api/pyspark/validate")
def validate(req: RunRequest):
    issues=[]
    code=req.code
    if "result" not in code:
        issues.append("Create the final DataFrame in a variable named result.")
    for table in next((q["tables"] for q in QUESTIONS if q["id"]==req.question_id), []):
        if table not in code:
            issues.append(f"Reference the required DataFrame: {table}")
    return {"ok":not issues, "issues":issues}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
