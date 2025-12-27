import subprocess
import sys
import time

python_cmd = sys.executable

services = [
    {
        "name": "API Gateway",
        "cmd": [
            python_cmd,
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
    },
    {"name": "Brain Engine", "cmd": [python_cmd, "brain.py"]},
    {"name": "LLM Strategist", "cmd": [python_cmd, "llm_strategist.py"]},
    {"name": "Notifier", "cmd": [python_cmd, "notifier.py"]},
]

processes = []


def start_services():
    for service in services:
        print(f"🚀 Starting {service['name']}...")
        p = subprocess.Popen(service["cmd"])
        processes.append(p)


def stop_services():
    for p in processes:
        p.terminate()


if __name__ == "__main__":
    try:
        start_services()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_services()
        print("\n🛑 System Shutdown.")
