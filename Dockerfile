FROM python:3.9
RUN mkdir /app
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
EXPOSE 5000
CMD [ "python3", "./app.py" ]