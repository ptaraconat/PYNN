FROM python:3.9

WORKDIR /code

COPY ./requirements.txt ./
COPY ./sources ./
COPY ./tests ./
COPY ./T1_curvefitting.py ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python -m pytest -k tests/*"]