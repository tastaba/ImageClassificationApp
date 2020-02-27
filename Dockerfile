FROM python:3.7

WORKDIR /SampleApp_GCP

RUN pip install pandas scikit-learn flask gunicorn

ADD model.h10 model_cnn.json
ADD main.py main.py

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "main:app" ]