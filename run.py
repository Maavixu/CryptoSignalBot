#!/usr/bin/env python3
"""
run.py — Start CryptoSignalBot.
Usage:
    python run.py              # default port 8000
    python run.py --port 8080  # custom port
    python run.py --reload     # dev mode with auto-reload
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import uvicorn

def main():
    parser = argparse.ArgumentParser(description="CryptoSignalBot server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════╗
║        CryptoSignalBot v1.0.0             ║
║        Phase 1: Foundation Active         ║
╠═══════════════════════════════════════════╣
║  API:      http://localhost:{args.port:<5}          ║
║  Frontend: open frontend/index.html       ║
║  Kill:     touch STOP  (halt orders)      ║
╚═══════════════════════════════════════════╝
    """)

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )

if __name__ == "__main__":
    main()
