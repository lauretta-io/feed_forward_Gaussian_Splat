FROM python:3.12-slim

WORKDIR /opt/ariadne
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

ENTRYPOINT ["ariadne"]
CMD ["--help"]
