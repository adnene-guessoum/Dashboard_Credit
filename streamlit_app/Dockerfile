FROM python:3.11

WORKDIR /app

RUN python -m pip install poetry

RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" \
    -- -d -b /bin

COPY Taskfile.yaml .
COPY pyproject.toml ./
COPY poetry.lock ./

RUN task install

COPY . .

EXPOSE 3000

CMD ["poetry", "run", "streamlit", "run", "app.py"]
