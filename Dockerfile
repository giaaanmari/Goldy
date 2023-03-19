FROM python:3.9

RUN pip install --upgrade pip

WORKDIR /docker-flask

ADD . /docker-flask

RUN pip install -r requirements.txt

CMD ["python", "app.py"]