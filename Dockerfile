FROM python:3.13
WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml ./
COPY requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy source code
COPY review_analysis/ ./review_analysis/
COPY model_stores/ ./model_stores/

EXPOSE 8000

CMD ["uvicorn", "review_analysis.api:app", "--host", "0.0.0.0", "--port", "8000"]