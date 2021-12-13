"""gunicorn server configuration."""
import os

threads = 2
workers = 1
timeout = 60
bind = f":{os.environ.get('PORT', '8080')}"
worker_class = "uvicorn.workers.UvicornWorker"
