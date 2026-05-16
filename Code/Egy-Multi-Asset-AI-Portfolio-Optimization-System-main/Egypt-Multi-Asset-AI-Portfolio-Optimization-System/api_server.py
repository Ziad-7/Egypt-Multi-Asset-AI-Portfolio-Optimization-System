from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from pathlib import Path
import sys

# Add the project root to sys.path to allow imports from src
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.api.service import run_portfolio_intelligence

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORT_PATH = Path("outputs/intelligence_report.json")

class RunOptions(BaseModel):
    include_backtest: bool = True

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/report/latest")
async def get_latest_report():
    if not REPORT_PATH.exists():
        return {"error": "Report not found. Please run the pipeline first."}
    try:
        with open(REPORT_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to read report: {str(e)}"}

@app.post("/api/report/run")
async def run_pipeline(options: RunOptions):
    try:
        result = run_portfolio_intelligence(include_backtest=options.include_backtest)
        return result
    except Exception as e:
        return {"error": f"Pipeline run failed: {str(e)}"}

@app.post("/api/report/run-and-save")
async def run_and_save_pipeline(options: RunOptions):
    try:
        result = run_portfolio_intelligence(include_backtest=options.include_backtest)
        # Guard against accidentally wiping chart data when saving a run
        # executed with include_backtest=False.
        if not options.include_backtest:
            prev = None
            if REPORT_PATH.exists():
                try:
                    with open(REPORT_PATH, "r") as f_prev:
                        prev = json.load(f_prev)
                except Exception:
                    prev = None
            prev_backtest = prev.get("backtest") if isinstance(prev, dict) else None
            new_backtest = result.get("backtest") if isinstance(result, dict) else None
            if isinstance(prev_backtest, dict) and prev_backtest and (not isinstance(new_backtest, dict) or not new_backtest):
                result["backtest"] = prev_backtest

        os.makedirs(REPORT_PATH.parent, exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        return result
    except Exception as e:
        return {"error": f"Pipeline run and save failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8787, reload=True)
