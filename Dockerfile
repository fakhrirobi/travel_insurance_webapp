FROM python:3.8.5
WORKDIR travel_insurance_webapp
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8051
CMD ["streamlit","run","webapp.py"]

