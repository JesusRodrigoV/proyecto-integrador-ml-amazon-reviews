#!/usr/bin/env python3
"""
Uso:
    python src/start_mlflow.py

Inicia MLflow tracking server + ngrok tunnel.
Requiere: mlflow instalado en el venv, ngrok CLI en PATH y autenticado.

Detener: Ctrl+C
"""

import socket
import subprocess
import time
import json
import os
import signal
import sys
import webbrowser
from pathlib import Path
from urllib.request import urlopen, Request

MLFLOW_HOME = Path(__file__).resolve().parent.parent / "mlflow"
MLFLOW_HOME.mkdir(parents=True, exist_ok=True)
DB_PATH = MLFLOW_HOME / "mlflow.db"
ARTIFACTS = MLFLOW_HOME / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
LOG_PATH = MLFLOW_HOME / "server.log"

PORT = 5000
processes = []


def wait_for_port(host="127.0.0.1", port=PORT, timeout=15):
    for i in range(timeout):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect((host, port))
            s.close()
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    return False


def start_mlflow():
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--backend-store-uri", f"sqlite:///{DB_PATH}",
        "--default-artifact-root", str(ARTIFACTS),
    ]
    log = open(LOG_PATH, "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    processes.append(p)
    time.sleep(2)
    if p.poll() is not None:
        print("  [ERROR] MLflow server no arrancó. Revisá:")
        print(f"          {LOG_PATH}")
        sys.exit(1)
    if not wait_for_port():
        print("  [ERROR] MLflow server arrancó pero no abre el puerto. Revisá:")
        print(f"          {LOG_PATH}")
        sys.exit(1)
    print(f"  [MLflow] server corriendo en :{PORT}")


def get_ngrok_url():
    for attempt in range(5):
        time.sleep(1)
        try:
            req = Request("http://localhost:4040/api/tunnels")
            data = json.loads(urlopen(req).read())
            for tunnel in data["tunnels"]:
                if tunnel["proto"] == "https":
                    return tunnel["public_url"]
        except Exception:
            continue
    return None


def start_ngrok():
    p = subprocess.Popen(
        ["ngrok", "http", f"http://127.0.0.1:{PORT}", "--host-header=rewrite"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    processes.append(p)

    url = get_ngrok_url()
    if url:
        print(f"  [Ngrok]   expuesto en {url}")
        return url
    else:
        print("  [ERROR] ngrok no respondió. Ejecutá:")
        print("          ngrok authtoken <TU_TOKEN>")
        return None


def cleanup(signum=None, frame=None):
    print("\nDeteniendo...")
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=3)
        except Exception:
            p.kill()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    print("\n  Iniciando MLflow tracking server + ngrok...\n")
    start_mlflow()
    url = start_ngrok()
    if url:
        print(f"\n  {'=' * 55}")
        print(f"  MLFLOW_TRACKING_URI = '{url}'")
        print(f"  {'=' * 55}")
        print(f"\n  Dashboard: {url}/#/experiments/0")
        print(f"  DB local:  {DB_PATH}")
        print(f"  Artifacts: {ARTIFACTS}")
        print(f"\n  Ctrl+C para detener.\n")
        webbrowser.open(f"{url}/#/experiments/0")
    else:
        print("\n  Sin ngrok. MLflow solo accesible via localhost.")
        print(f"  MLFLOW_TRACKING_URI = 'http://localhost:{PORT}'")
        print(f"\n  Ctrl+C para detener.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()
