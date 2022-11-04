FROM webis/pan-clickbait-spoiling-baselines:task1-transformer-0.0.3

RUN git+https://github.com/ibot3/MILES.git

COPY simplifier /

ENTRYPOINT [ "/miles-transformer-task-1.py" ]

