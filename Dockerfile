FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY . /tmp/EnsembleOT

RUN pip install "/tmp/EnsembleOT[dev]" \
    && cd /tmp/EnsembleOT \
    && pytest -q \
    && rm -rf /tmp/EnsembleOT

CMD ["python"]
