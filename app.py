import ast
import json
import os
import shutil
import subprocess
import threading
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Spark / Java configuration
# These must be configured before importing PySpark.
# ---------------------------------------------------------------------------
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
os.environ.setdefault("PYSPARK_PYTHON", "python3")
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "python3")


def detect_java_home():
    """Find JAVA_HOME or an installed Java runtime."""
    configured = os.environ.get("JAVA_HOME")
    candidates = []

    if configured:
        candidates.append(Path(configured))

    java = shutil.which("java")
    if java:
        candidates.append(Path(java).resolve().parent.parent)

    candidates.extend(
        [
            Path("/usr/lib/jvm/java-17-openjdk-amd64"),
            Path("/usr/lib/jvm/java-17-openjdk"),
            Path("/usr/lib/jvm/java-11-openjdk-amd64"),
            Path("/usr/lib/jvm/java-11-openjdk"),
        ]
    )

    for home in candidates:
        if (home / "bin" / "java").exists():
            os.environ["JAVA_HOME"] = str(home)
            return str(home)

    return None


JAVA_HOME_FOUND = detect_java_home()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent

try:
    DATA = json.loads((BASE / "data.json").read_text(encoding="utf-8"))
    QUESTIONS = json.loads(
        (BASE / "questions.json").read_text(encoding="utf-8")
    )
except Exception as exc:
    raise RuntimeError(f"Could not load data.json/questions.json: {exc}") from exc


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SQL Interview Practice - PySpark Backend",
    version="2.0.1",
)

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


# ---------------------------------------------------------------------------
# Spark state
# ---------------------------------------------------------------------------
spark_lock = threading.Lock()
spark = None
frames = {}
spark_error = None


def java_version():
    java = shutil.which("java")

    if not java:
        return None

    try:
        process = subprocess.run(
            [java, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        text = (process.stderr or process.stdout or "").splitlines()
        return text[0] if text else "unknown"
    except Exception:
        return "unknown"


def get_spark():
    global spark, frames, spark_error

    if spark is not None:
        return spark

    if not JAVA_HOME_FOUND:
        spark_error = (
            "Java was not found. Install OpenJDK 17 and set JAVA_HOME."
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
            frames[name] = spark.createDataFrame(
                meta["rows"],
                meta["columns"],
            )

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


def clean_value(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
@app.get("/api/question/{question_id}")
def question(question_id: str):
    q = next((item for item in QUESTIONS if item["id"] == question_id), None)

    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    return {
        "id": q["id"],
        "title": q["title"],
        "difficulty": q["difficulty"],
        "tables": q["tables"],
        "expected": q["expected"],
        "columns": q["columns"],
    }


# ---------------------------------------------------------------------------
# PySpark practice-code validation
# ---------------------------------------------------------------------------
BLOCKED_TEXT = [
    "os.system",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "requests",
    "urllib",
    "httpx",
    "open(",
    "input(",
    "eval(",
    "exec(",
    "compile(",
    "__import__",
    "globals(",
    "locals(",
    "breakpoint(",
    "importlib",
]

ALLOWED_IMPORTS = {
    "pyspark",
    "pyspark.sql",
    "pyspark.sql.functions",
    "pyspark.sql.window",
}

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


def validate_user_code(code: str):
    normalized = code.lower().replace(" ", "")
    hits = [
        item
        for item in BLOCKED_TEXT
        if item.lower().replace(" ", "") in normalized
    ]

    if hits:
        raise HTTPException(
            status_code=400,
            detail="Blocked operation in practice code: "
            + ", ".join(hits),
        )

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Python syntax error: {exc}",
        ) from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [
                alias.name
                for alias in node.names
                if alias.name not in ALLOWED_IMPORTS
            ]
            if bad:
                raise HTTPException(
                    status_code=400,
                    detail="Only PySpark imports are allowed in practice code.",
                )

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module not in ALLOWED_IMPORTS:
                raise HTTPException(
                    status_code=400,
                    detail="Only PySpark imports are allowed in practice code.",
                )


def run_user_code(code: str):
    """Execute validated PySpark practice code."""
    validate_user_code(code)

    env = {
        "spark": spark,
        "F": F,
        **frames,
    }

    globals_dict = {
        "__builtins__": SAFE_BUILTINS,
    }

    exec(code, globals_dict, env)

    return env.get("result")


# ---------------------------------------------------------------------------
# Execute PySpark
# ---------------------------------------------------------------------------
@app.post("/api/pyspark/run")
def run_pyspark(req: RunRequest):
    q = next((item for item in QUESTIONS if item["id"] == req.question_id), None)

    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    with spark_lock:
        try:
            get_spark()

            result = run_user_code(req.code)

            if result is None or not hasattr(result, "collect"):
                return {
                    "ok": False,
                    "error": (
                        "Your code must create the final DataFrame "
                        "in a variable named result."
                    ),
                }

            rows = result.collect()
            cols = result.columns

            payload = [
                [clean_value(row[column]) for column in cols]
                for row in rows
            ]

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
                    else (
                        "Code executed successfully, but the result "
                        "does not match the expected output."
                    )
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


# ---------------------------------------------------------------------------
# Validate PySpark structure without executing Spark
# ---------------------------------------------------------------------------
@app.post("/api/pyspark/validate")
def validate(req: RunRequest):
    issues = []
    code = req.code

    try:
        validate_user_code(code)
    except HTTPException as exc:
        return {
            "ok": False,
            "issues": [str(exc.detail)],
        }

    if "result" not in code:
        issues.append(
            "Create the final DataFrame in a variable named result."
        )

    question = next(
        (item for item in QUESTIONS if item["id"] == req.question_id),
        None,
    )

    if question:
        for table in question["tables"]:
            if table not in code:
                issues.append(
                    f"Reference the required DataFrame: {table}"
                )

    return {
        "ok": not issues,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
