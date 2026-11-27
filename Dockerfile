FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_URI=/app/models/RentalPricePredictor_champion_v1

COPY models/RentalPricePredictor_champion_v1/requirements.txt model-requirements.txt

RUN pip install --no-cache-dir \
    -r model-requirements.txt \
    fastapi==0.165.0 \
    uvicorn==0.54.0

COPY src/ ./src/

COPY models/RentalPricePredictor_champion_v1/ \
    ./models/RentalPricePredictor_champion_v1/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.mp172.api:app", "--host", "0.0.0.0", "--port", "8000"]