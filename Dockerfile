FROM python:3.9

WORKDIR /code

COPY ./requirements.txt ./
COPY ./sources ./
COPY ./tests ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["pytest", "tests/"]