"""Cloud Run Function entry point for the weather paper trade collector."""

import functions_framework
from flask import Request

from collector import collect


@functions_framework.http
def run_collector(request: Request):
    """HTTP entry point triggered daily by Cloud Scheduler."""
    try:
        collect()
        return {"status": "ok"}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500
