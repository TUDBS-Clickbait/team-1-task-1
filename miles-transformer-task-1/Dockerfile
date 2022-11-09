FROM webis/pan-clickbait-spoiling-baselines:task1-transformer-0.0.3

RUN pip3 install git+https://github.com/ibot3/MILES.git

COPY simplifier/ /simplifier
COPY ../data/ /data

ENTRYPOINT [ "/miles-transformer-task-1.py" ]

