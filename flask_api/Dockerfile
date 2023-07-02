FROM python:3.11 as builder

WORKDIR /app

RUN python -m pip install poetry

RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" \
    -- -d -b /bin
COPY ../Taskfile.yaml .

COPY ../pyproject.toml ./
COPY ../poetry.lock ./
RUN task install

COPY . .

CMD ["poetry", "run", "flask", "run"]
