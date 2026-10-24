# Container for DigitalOcean App Platform (or any Docker host)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py LICENSE README.md ./
COPY .streamlit ./.streamlit
ENV PORT=8080
EXPOSE 8080
# The model key is provided as the DO_MODEL_ACCESS_KEY environment variable (encrypted secret on App Platform)
CMD streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true \
    --server.enableCORS false --server.enableXsrfProtection false
