FROM python:3.10-slim-bookworm
RUN useradd -m agent && mkdir -p /task-project /fixture /installed-agent/iris
COPY fake_provider.py /fixture/fake_provider.py
COPY sitecustomize.py /installed-agent/iris/sitecustomize.py
COPY scenario.json /fixture/scenario.json
COPY input.csv pixel.png /task-project/
RUN chown -R agent:agent /task-project
WORKDIR /task-project
ENTRYPOINT ["python", "/fixture/fake_provider.py"]
