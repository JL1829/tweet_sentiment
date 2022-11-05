FROM python:3.7.15
LABEL maintainer="Lu ZhiPing"

USER root
SHELL [ "/bin/bash", "-c" ]

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt

COPY tokenizer /opt/tokenizer
COPY MLEngineBert.py /opt
COPY torch-model.onnx /opt
COPY app.py /opt
COPY requirements.txt /opt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --timeout 10000 -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
