FROM python:3.9

# Set environment variables for configuration and defaults
ENV TEST_DIR=/app/tests
ENV SOURCE_DIR=/app/sources
ENV TUTO_DIR=/app/tutorials
ENV TEST_FILE=test_layers.py.py
# Set the working directory inside the container
WORKDIR /app

# Copy the test files and sources into the container
COPY tests $TEST_DIR
COPY sources $SOURCE_DIR
COPY tutorials $TUTO_DIR
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pytest

CMD pytest tests
#CMD ["python", "tutorials/T1_curvefitting.py"]