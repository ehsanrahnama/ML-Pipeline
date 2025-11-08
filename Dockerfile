FROM python:3.10-slim


WORKDIR /app

COPY ./requirements.txt .
# COPY setup.py .
COPY ./src src/ 

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e . 


# EXPOSE 8000

# Command to run FastAPI app
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]