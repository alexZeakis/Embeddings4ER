FROM python:3.9
WORKDIR /app
COPY . /app/
COPY ./reproducibility/run_all.sh /app/run_all.sh
RUN apt-get update && apt-get -y install rsync
RUN mv /app/logs/ /app/or_logs/
RUN pip install --no-cache-dir -r requirements.txt
CMD ["tail", "-f", "/dev/null"]
