import ast
import json
import os
import shutil
import subprocess
import threading
import traceback
from pathlib import Path

# Spark/Java settings must be configured BEFORE importing PySpark.
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
os.environ.setdefault("PYSPARK_PYTHON", "python3")
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "python3")


def detect_java_home():
    """Find Java 17 (preferred) or another installed Java runtime."""
    configured = os.environ.get("JAVA_HOME")
    candidates = []
    if configured:
        candidates.append(Path(configured))

    java = shutil.which("java")
    if java:
        candidates.append(Path(java).resolve().parent.parent)

    candidates.extend([
        Path("/usr/lib/jvm/java-17-openjdk-amd64"),
        Path("/usr/lib/jvm/java-17-openjdk"),
        Path("/usr/lib/jvm/java-11-openjdk-amd64"),
        Path("/usr/lib/jvm/java-11-openjdk"),
    ])

    for home in candidates:
        if (home / "bin" / "java").exists():
            os.environ["JAVA_HOME"] = str(home)
            return str(home)
    return None


JAVA_HOME_FOUND = detect_java_home()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE = Path(__file__).resolve().parent
DATA = json.loads((BASE / "data.json").read_text(encoding="utf-8"))
QUESTIONS = json.loads((BASE / "questions.json").read_text(encoding="utf-8"))

app = FastAPI(title="SQL Interview Practice - PySpark Backend", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    question_id: str = Field(min_length=1)
    code: str = Field(min_length=1, max_length=50000)


spark_lock = threading.Lock()
spark = None
frames = {}
spark_error = None


def java_version():
    java = shutil.which("java")
    if not java:
        return None
    try:
        p = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=5)
        text = (p.stderr or p.stdout or "").splitlines()
        return text[0] if text else "unknown"
    except Exception:
        return "unknown"


def get_spark():
    global spark, frames, spark_error

    if spark is not None:
        return spark

    if not JAVA_HOME_FOUND:
        spark_error = (
            "Java was not found. Render must install OpenJDK 17 before starting PySpark. "
            "Set JAVA_HOME to the Java 17 installation."
        )
        raise RuntimeError(spark_error)

    try:
        spark = (
            SparkSession.builder
            .master("local[1]")
            .appName("SQLInterviewPractice")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.default.parallelism", "2")
            .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
            .config("spark.driver.memory", "384m")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

        for name in DATA["tables"]:
            meta = DATA["data"][name]
            frames[name] = spark.createDataFrame(meta["rows"], meta["columns"])

        spark_error = None
        return spark
    except Exception as exc:
        spark = None
        frames = {}
        spark_error = str(exc)
        raise RuntimeError(
            "Spark could not start. "
            f"JAVA_HOME={os.environ.get('JAVA_HOME', 'not set')}; "
            f"Java={java_version() or 'not found'}. "
            f"Original error: {exc}"
        ) from exc


def clean_value(v):
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "spark": "ready" if spark is not None else "not_started",
        "spark_error": spark_error,
        "java_home": os.environ.get("JAVA_HOME"),
        "java": java_version(),
        "tables": len(DATA["tables"]),
        "questions": len(QUESTIONS),
    }


@app.get("/api/question/{question_id}")
def question(question_id: str):
    q = next((x for x in QUESTIONS if x["id"] == question_id), None)
    if not q:
        raise HTTPException(404, "Question not found")
    return {
        "id": q["id"],
        "title": q["title"],
        "difficulty": q["difficulty"],
        "tables": q["tables"],
        "expected": q["expected"],
        "columns": q["columns"],
    }


# This is a practice service, not a general Python sandbox.
# Keep the public endpoint limited to Spark/DataFrame operations.
BLOCKED_TEXT = [
    "os.system", "subprocess", "socket", "shutil", "pathlib", "requests",
    "urllib", "httpx", "open(", "input(", "eval(", "exec(", "compile(",
    "__import__", "globals(", "locals(", "breakpoint(", "importlib",
]
ALLOWED_IMPORTS = {
    "pyspark",
    "pyspark.sql",
    "pyspark.sql.functions",
    "pyspark.sql.window",
}
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "True": True, "False": False, "None": None,
}


def validate_user_code(code: str):
    low = code.lower().replace(" ", "")
    hits = [x for x in BLOCKED_TEXT if x.lower().replace(" ", "") in low]
    if hits:
        raise HTTPException(400, "Blocked operation in practice code: " + ", ".join(hits))

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise HTTPException(400, f"Python syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module or ""
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
                bad = [n for n in names if n not in ALLOWED_IMPORTS]
            else:
                bad = [] if module in ALLOWED_IMPORTS else [module]
            if bad:
                raise HTTPException(400, "Only PySpark imports are allowed in practice code.")


def run_user_code(code: str):
    """Execute only after static validation, with a reduced builtin set."""
    validate_user_code(code)
    env = {"spark": spark, "F": F, **frames}
    globals_dict = {"__builtins__": SAFE_BUILTINS}
    exec(code, globals_dict, env)
    return env.get("result")


@app.post("/api/pyspark/run")
def run_pyspark(req: RunRequest):
    q = next((x for x in QUESTIONS if x["id"] == req.question_id), None)
    if not q:
        raise HTTPException(404, "Question not found")

    with spark_lock:
        try:
            get_spark()
            result = run_user_code(req.code)

            if result is None or not hasattr(result, "collect"):
                return {
                    "ok": False,
                    "error": "Your code must create the final DataFrame in a variable named result.",
                }

            rows = result.collect()
            cols = result.columns
            payload = [[clean_value(r[c]) for c in cols] for r in rows]
            expected = q.get("expected", [])
            matched = payload == expected

            return {
                "ok": True,
                "matched": matched,
                "columns": cols,
                "rows": payload[:500],
                "row_count": len(payload),
                "expected_columns": q.get("columns", []),
                "expected_rows": expected,
                "message": (
                    "Correct answer"
                    if matched
                    else "Code executed successfully, but the result does not match the expected output."
                ),
            }
        except HTTPException:
            raise
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }


@app.post("/api/pyspark/validate")
def validate(req: RunRequest):
    issues = []
    code = req.code
    if "result" not in code:
        issues.append("Create the final DataFrame in a variable named result.")
    for table in next((q["tables"] for q in QUESTIONS if q["id"] == req.question_id), []):
        if table not in code:
            issues.append(f"Reference the required DataFrame: {table}")
    return {"ok": not issues, "issues": issues}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)
