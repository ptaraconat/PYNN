FROM python:3.9

WORKDIR /code

COPY ./requirements.txt ./
COPY ./sources ./
COPY ./T1_curvefitting.py ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "T1_curvefitting.py"]