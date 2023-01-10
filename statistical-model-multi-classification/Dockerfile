FROM python:3.10

RUN apt-get update && apt-get install -y git-lfs wget 

RUN mkdir /opt/app

RUN git clone 'https://huggingface.co/webis/spoiler-type-classification' /opt/app/non-multi-model 

WORKDIR /opt/app/non-multi-model
RUN git lfs install 
RUN git fetch 
RUN git checkout --track origin/deberta-all-three-types-concat-1-checkpoint-1000-epoch-10 
RUN rm -Rf .git

WORKDIR /opt/app/

COPY requirements.txt /opt/app/

RUN pip3 install -r requirements.txt

COPY multi.model multipart_detection.py two-step-transformer.py /opt/app/

ENTRYPOINT [ "/opt/app/two-step-transformer.py" ]
