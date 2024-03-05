FROM python:3

WORKDIR /urs/src/app
COPY . .
RUN pip install numpy && pip install pandas && pip install sklearn && pip install matplotlib
CMD [ "python", "-u","./server.py"]