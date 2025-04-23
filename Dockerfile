FROM python:3.8
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model.tflite .
COPY birdnames.db .
COPY speciesid.py .
COPY webui.py .
COPY queries.py .
COPY api.py .
COPY file_watcher.py .
COPY templates/ ./templates/
COPY static/ ./static/
COPY ./config/ ./config/

EXPOSE 8000
# Use environment variable to determine which app to run
ENV APP_TYPE="api"
CMD if [ "$APP_TYPE" = "api" ]; then \
    python ./api.py; \
    elif [ "$APP_TYPE" = "file_watcher" ]; then \
    python ./file_watcher.py; \
    else \
    python ./speciesid.py; \
    fi
