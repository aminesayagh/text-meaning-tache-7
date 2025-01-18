# Code Documentation
Generated on: 2025-01-18T20:37:56.274Z
Total files: 15

## Project Structure

```
└── Tache7
    ├── doc
    │   └── cw.sh
    ├── docker-compose.yml
    ├── grafana
    │   └── provisioning
    │       ├── dashboards
    │       │   └── dashboards.yml
    │       └── datasources
    │           └── datasource.yml
    ├── models
    │   ├── models.config
    │   └── monitoring.config
    ├── nginx.conf
    ├── prometheus
    │   └── prometheus.yml
    ├── scripts
    │   ├── kafka
    │   │   ├── init-topics.sh
    │   │   └── text_producer.py
    │   ├── spark
    │   │   ├── jobs
    │   │   │   └── text_processor.py
    │   │   └── submit-job.sh
    │   └── utils
    │       ├── model_client.py
    │       └── s3Handler.py
    └── spark-defaults.conf
```

## File: docker-compose.yml
- Path: `/root/git/text-meaning/Tache7/docker-compose.yml`
- Size: 6.21 KB
- Extension: .yml
- Lines of code: 208

```yml
services:
zookeeper:
image: confluentinc/cp-zookeeper:7.3.0
container_name: zookeeper
ports:
- "2181:2181"
environment:
ZOOKEEPER_CLIENT_PORT: 2181
ZOOKEEPER_TICK_TIME: 2000
kafka:
image: confluentinc/cp-kafka:7.3.0
container_name: kafka
depends_on:
- zookeeper
ports:
- "9092:9092"
environment:
KAFKA_BROKER_ID: 1
KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
volumes:
- ../scripts/kafka:/scripts
healthcheck:
test: ["CMD-SHELL", "kafka-topics --bootstrap-server localhost:9092 --list"]
interval: 30s
timeout: 10s
retries: 3
kafka-init:
image: confluentinc/cp-kafka:7.3.0
depends_on:
kafka:
condition: service_healthy
volumes:
- ../scripts/kafka:/scripts
command: /scripts/init-topics.sh
kafka-ui:
image: provectuslabs/kafka-ui:latest
container_name: kafka-ui
depends_on:
- kafka
ports:
- "8080:8080"
environment:
KAFKA_CLUSTERS_0_NAME: local
KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:29092
KAFKA_CLUSTERS_0_ZOOKEEPER: zookeeper:2181
spark-master:
image: bitnami/spark:3.3.2
container_name: spark-master
environment:
- SPARK_MODE=master
- SPARK_RPC_AUTHENTICATION_ENABLED=no
- SPARK_RPC_ENCRYPTION_ENABLED=no
- SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
- SPARK_SSL_ENABLED=no
- SPARK_MASTER_URL=spark://spark-master:7077
- SPARK_DRIVER_MEMORY=2g
- SPARK_EXECUTOR_MEMORY=2g
ports:
- "8081:8080"  # Web UI
- "7077:7077"  # Master port
- "4040:4040"  # Spark application UI
volumes:
- ../scripts/spark:/opt/spark-apps
- ../data:/opt/spark-data
- ./spark-defaults.conf:/opt/bitnami/spark/conf/spark-defaults.conf
healthcheck:
test: ["CMD", "curl", "-f", "http://localhost:8080"]
interval: 5s
timeout: 3s
retries: 3
networks:
- text-mining-network
spark-worker-1:
image: bitnami/spark:3.3.2
container_name: spark-worker-1
environment:
- SPARK_MODE=worker
- SPARK_MASTER_URL=spark://spark-master:7077
- SPARK_WORKER_MEMORY=2g
- SPARK_WORKER_CORES=2
- SPARK_RPC_AUTHENTICATION_ENABLED=no
- SPARK_RPC_ENCRYPTION_ENABLED=no
- SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
- SPARK_SSL_ENABLED=no
volumes:
- ../scripts/spark:/opt/spark-apps
- ../data:/opt/spark-data
- ./spark-defaults.conf:/opt/bitnami/spark/conf/spark-defaults.conf
depends_on:
spark-master:
condition: service_healthy
networks:
- text-mining-network
spark-worker-2:
image: bitnami/spark:3.3.2
container_name: spark-worker-2
environment:
- SPARK_MODE=worker
- SPARK_MASTER_URL=spark://spark-master:7077
- SPARK_WORKER_MEMORY=2g
- SPARK_WORKER_CORES=2
- SPARK_RPC_AUTHENTICATION_ENABLED=no
- SPARK_RPC_ENCRYPTION_ENABLED=no
- SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
- SPARK_SSL_ENABLED=no
volumes:
- ../scripts/spark:/opt/spark-apps
- ../data:/opt/spark-data
- ./spark-defaults.conf:/opt/bitnami/spark/conf/spark-defaults.conf
depends_on:
spark-master:
condition: service_healthy
networks:
- text-mining-network
s3-init:
image: amazon/aws-cli:latest
environment:
- AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
- AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
- AWS_REGION=${AWS_REGION}
command: >
/bin/sh -c "
aws s3api create-bucket --bucket ${S3_RAW_BUCKET} --region ${AWS_REGION} || true;
aws s3api create-bucket --bucket ${S3_PROCESSED_BUCKET} --region ${AWS_REGION} || true;
aws s3api create-bucket --bucket ${S3_MODELS_BUCKET} --region ${AWS_REGION} || true;
"
tensorflow-serving:
image: tensorflow/serving:latest
container_name: tf-serving
ports:
- "8501:8501"  # REST API
- "8500:8500"  # gRPC
volumes:
- ../models:/models
environment:
- MODEL_NAME=text_classifier
- MODEL_BASE_PATH=/models/text_classifier
- TENSORFLOW_INTER_OP_PARALLELISM=4
- TENSORFLOW_INTRA_OP_PARALLELISM=4
command:
- "--model_config_file=/models/models.config"
- "--monitoring_config_file=/models/monitoring.config"
- "--enable_batching=true"
- "--rest_api_timeout_in_ms=30000"
- "--tensorflow_session_parallelism=4"
healthcheck:
test: ["CMD", "curl", "-f", "http://localhost:8501/v1/models/text_classifier"]
interval: 30s
timeout: 10s
retries: 3
networks:
- text-mining-network
tensorflow-serving-proxy:
image: nginx:alpine
container_name: tf-proxy
ports:
- "8502:80"
volumes:
- ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
depends_on:
tensorflow-serving:
condition: service_healthy
networks:
- text-mining-network
prometheus:
image: prom/prometheus:latest
container_name: prometheus
ports:
- "9090:9090"
volumes:
- ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
- prometheus_data:/prometheus
command:
- '--config.file=/etc/prometheus/prometheus.yml'
- '--storage.tsdb.path=/prometheus'
- '--web.console.libraries=/usr/share/prometheus/console_libraries'
- '--web.console.templates=/usr/share/prometheus/consoles'
networks:
- text-mining-network
grafana:
image: grafana/grafana:latest
container_name: grafana
ports:
- "3000:3000"
environment:
- GF_SECURITY_ADMIN_PASSWORD=admin
- GF_SECURITY_ADMIN_USER=admin
- GF_USERS_ALLOW_SIGN_UP=false
volumes:
- ./grafana/provisioning:/etc/grafana/provisioning
- grafana_data:/var/lib/grafana
depends_on:
- prometheus
networks:
- text-mining-network
networks:
text-mining-network:
driver: bridge
volumes:
prometheus_data:
grafana_data:
kafka_data:
spark_data:
model_data:
```

---------------------------------------------------------------------------

## File: nginx.conf
- Path: `/root/git/text-meaning/Tache7/nginx.conf`
- Size: 952.00 B
- Extension: .conf
- Lines of code: 33

```conf
events {
worker_connections 1024;
}
http {
upstream tensorflow-serving {
server tensorflow-serving:8501;
}
server {
listen 80;
location / {
proxy_pass http://tensorflow-serving;
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
# Timeouts
proxy_connect_timeout 60s;
proxy_send_timeout 60s;
proxy_read_timeout 60s;
# Buffer settings
proxy_buffering on;
proxy_buffer_size 128k;
proxy_buffers 4 256k;
proxy_busy_buffers_size 256k;
# HTTP/1.1 support
proxy_http_version 1.1;
proxy_set_header Connection "";
}
# Health check endpoint
location /health {
access_log off;
return 200 'healthy\n';
}
}
}
```

---------------------------------------------------------------------------

## File: spark-defaults.conf
- Path: `/root/git/text-meaning/Tache7/spark-defaults.conf`
- Size: 1.93 KB
- Extension: .conf
- Lines of code: 47

```conf
# /Tache7/docker/spark-defaults.conf
# Application Settings
spark.app.name                             text-mining-pipeline
# Memory Settings
spark.driver.memory                        2g
spark.executor.memory                      2g
spark.driver.maxResultSize                 1g
# Execution Settings
spark.executor.instances                   2
spark.executor.cores                       2
spark.default.parallelism                  4
# Hadoop AWS Integration
spark.hadoop.fs.s3a.impl                   org.apache.hadoop.fs.s3a.S3AFileSystem
spark.hadoop.fs.s3a.endpoint              s3.amazonaws.com
spark.hadoop.fs.s3a.fast.upload            true
spark.hadoop.fs.s3a.connection.maximum     100
spark.hadoop.fs.s3a.connection.timeout     200000
spark.hadoop.fs.s3a.attempts.maximum       20
spark.hadoop.fs.s3a.multipart.size        104857600
spark.hadoop.fs.s3a.multipart.threshold    104857600
# Serialization
spark.serializer                          org.apache.spark.serializer.KryoSerializer
spark.kryoserializer.buffer.max           1024m
# Shuffle Settings
spark.shuffle.file.buffer                 32k
spark.shuffle.compress                    true
spark.shuffle.spill.compress             true
# Memory Management
spark.memory.fraction                     0.6
spark.memory.storageFraction             0.5
# IO Settings
spark.io.compression.codec                lz4
spark.io.compression.lz4.blockSize        32k
# Network Settings
spark.network.timeout                     120s
spark.executor.heartbeatInterval         10s
# Performance Tuning
spark.sql.files.maxPartitionBytes         134217728
spark.sql.shuffle.partitions              10
spark.default.parallelism                 10
# Memory Settings
spark.memory.fraction                     0.8
spark.memory.storageFraction             0.3
# Streaming Settings
spark.streaming.backpressure.enabled      true
spark.streaming.kafka.maxRatePerPartition 100
spark.streaming.stopGracefullyOnShutdown  true
```

---------------------------------------------------------------------------

## File: cw.sh
- Path: `/root/git/text-meaning/Tache7/doc/cw.sh`
- Size: 526.00 B
- Extension: .sh
- Lines of code: 18

```sh
#!/bin/bash
# Create docs directory if it doesn't exist
mkdir -p docs
# Generate tree structure with specific ignores
TREE_OUTPUT=$(tree -a -I 'node_modules|.git|.next|dist|.turbo|.cache|.vercel|coverage' \
--dirsfirst \
--charset=ascii)
{
echo "# Project Tree Structure"
echo "\`\`\`plaintext"
echo "$TREE_OUTPUT"
echo "\`\`\`"
} > docs/doc-project-tree.md
cw doc \
--pattern ".yml|.conf|.json|.sh|.py" \
--exclude ".pyc|.txt|.ipynb|.ipynb" \
--output "docs/doc.md" \
--compress false
```

---------------------------------------------------------------------------

## File: models.config
- Path: `/root/git/text-meaning/Tache7/models/models.config`
- Size: 365.00 B
- Extension: .config
- Lines of code: 21

```config
model_config_list {
config {
name: "text_classifier"
base_path: "/models/text_classifier"
model_platform: "tensorflow"
model_version_policy {
specific {
versions: 1
versions: 2
}
}
version_labels {
key: "production"
value: 2
}
version_labels {
key: "staging"
value: 1
}
}
}
```

---------------------------------------------------------------------------

## File: monitoring.config
- Path: `/root/git/text-meaning/Tache7/models/monitoring.config`
- Size: 187.00 B
- Extension: .config
- Lines of code: 9

```config
metrics_collector_config {
use_prometheus: true
prometheus_config {
port: 9090
enable_metric_label: true
}
collection_interval_ms: 1000
allowed_metric_name_regex: ".*"
}
```

---------------------------------------------------------------------------

## File: prometheus.yml
- Path: `/root/git/text-meaning/Tache7/prometheus/prometheus.yml`
- Size: 518.00 B
- Extension: .yml
- Lines of code: 19

```yml
global:
scrape_interval: 15s
evaluation_interval: 15s
scrape_configs:
- job_name: 'spark'
static_configs:
- targets: ['spark-master:8080']
metrics_path: /metrics
- job_name: 'kafka'
static_configs:
- targets: ['kafka:9092']
metrics_path: /metrics
- job_name: 'tensorflow-serving'
static_configs:
- targets: ['tensorflow-serving:9090']
metrics_path: /monitoring/prometheus/metrics
- job_name: 'prometheus'
static_configs:
- targets: ['localhost:9090']
```

---------------------------------------------------------------------------

## File: init-topics.sh
- Path: `/root/git/text-meaning/Tache7/scripts/kafka/init-topics.sh`
- Size: 652.00 B
- Extension: .sh
- Lines of code: 10

```sh
#!/bin/bash
# Wait for Kafka to be ready
sleep 10
# Create topics
kafka-topics --create --if-not-exists --bootstrap-server kafka:29092 --topic raw-text-fr --partitions 3 --replication-factor 1
kafka-topics --create --if-not-exists --bootstrap-server kafka:29092 --topic raw-text-en --partitions 3 --replication-factor 1
kafka-topics --create --if-not-exists --bootstrap-server kafka:29092 --topic raw-text-ar --partitions 3 --replication-factor 1
kafka-topics --create --if-not-exists --bootstrap-server kafka:29092 --topic processed-text --partitions 3 --replication-factor 1
# List created topics
kafka-topics --list --bootstrap-server kafka:29092
```

---------------------------------------------------------------------------

## File: text_producer.py
- Path: `/root/git/text-meaning/Tache7/scripts/kafka/text_producer.py`
- Size: 2.49 KB
- Extension: .py
- Lines of code: 68

```py
from kafka import KafkaProducer
import json
from typing import Dict, Any
import logging
class TextProducer:
def __init__(self, bootstrap_servers: str = 'kafka:29092'):
self.producer = KafkaProducer(
bootstrap_servers=bootstrap_servers,
value_serializer=lambda v: json.dumps(v).encode('utf-8'),
key_serializer=lambda k: k.encode('utf-8') if k else None
)
self.topics = {
'fr': 'raw-text-fr',
'en': 'raw-text-en',
'ar': 'raw-text-ar'
}
logging.basicConfig(level=logging.INFO)
self.logger = logging.getLogger(__name__)
def send_text(self, text: str, language: str, metadata: Dict[Any, Any] = None) -> None:
"""
Send text data to appropriate Kafka topic based on language
"""
if language not in self.topics:
raise ValueError(f"Unsupported language: {language}. Supported languages: {list(self.topics.keys())}")
topic = self.topics[language]
# Prepare the message
message = {
'text': text,
'language': language,
'metadata': metadata or {},
'timestamp': str(datetime.datetime.now())
}
try:
# Send the message
future = self.producer.send(topic, value=message, key=str(uuid.uuid4()))
# Wait for the message to be delivered
record_metadata = future.get(timeout=10)
self.logger.info(
f"Successfully sent message to topic {topic} "
f"partition {record_metadata.partition} "
f"offset {record_metadata.offset}"
)
except Exception as e:
self.logger.error(f"Error sending message to Kafka: {str(e)}")
raise
def close(self):
"""
Close the producer connection
"""
self.producer.close()
if __name__ == "__main__":
# Example usage
producer = TextProducer()
try:
# Example: Send a test message for each language
test_messages = {
'fr': 'Exemple de texte en français',
'en': 'Example text in English',
'ar': 'مثال نص باللغة العربية'
}
for lang, text in test_messages.items():
producer.send_text(
text=text,
language=lang,
metadata={'source': 'test', 'category': 'example'}
)
finally:
producer.close()
```

---------------------------------------------------------------------------

## File: submit-job.sh
- Path: `/root/git/text-meaning/Tache7/scripts/spark/submit-job.sh`
- Size: 627.00 B
- Extension: .sh
- Lines of code: 12

```sh
#!/bin/bash
spark-submit \
--master spark://spark-master:7077 \
--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2,org.apache.hadoop:hadoop-aws:3.3.2,com.amazonaws:aws-java-sdk-bundle:1.11.1026 \
--conf "spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem" \
--conf "spark.hadoop.fs.s3a.access.key=${AWS_ACCESS_KEY_ID}" \
--conf "spark.hadoop.fs.s3a.secret.key=${AWS_SECRET_ACCESS_KEY}" \
--conf "spark.hadoop.fs.s3a.endpoint=s3.amazonaws.com" \
--conf "spark.hadoop.fs.s3a.path.style.access=true" \
--driver-memory 2g \
--executor-memory 2g \
/opt/spark-apps/jobs/text_processor_s3.py
```

---------------------------------------------------------------------------

## File: model_client.py
- Path: `/root/git/text-meaning/Tache7/scripts/utils/model_client.py`
- Size: 2.57 KB
- Extension: .py
- Lines of code: 65

```py
import requests
import numpy as np
import json
from typing import List, Dict, Any
import logging
class ModelClient:
def __init__(self, host: str = "tensorflow-serving", port: int = 8501):
self.base_url = f"http://{host}:{port}/v1/models/text_classifier"
self.logger = logging.getLogger(__name__)
def predict(self, texts: List[str], version: str = "production") -> Dict[str, Any]:
"""
Make predictions using the TensorFlow Serving endpoint
"""
# Convert texts to model input format
instances = [{"text": text} for text in texts]
# Prepare request URL with version label
url = f"{self.base_url}/labels/{version}:predict"
try:
response = requests.post(url, json={"instances": instances})
response.raise_for_status()
predictions = response.json()
return predictions["predictions"]
except requests.exceptions.RequestException as e:
self.logger.error(f"Error making prediction request: {e}")
raise
def get_model_status(self) -> Dict[str, Any]:
"""
Get the status of the model serving
"""
try:
response = requests.get(f"{self.base_url}")
response.raise_for_status()
return response.json()
except requests.exceptions.RequestException as e:
self.logger.error(f"Error getting model status: {e}")
raise
def get_model_metadata(self, version: str = "production") -> Dict[str, Any]:
"""
Get model metadata for specific version
"""
try:
response = requests.get(f"{self.base_url}/labels/{version}/metadata")
response.raise_for_status()
return response.json()
except requests.exceptions.RequestException as e:
self.logger.error(f"Error getting model metadata: {e}")
raise
if __name__ == "__main__":
# Example usage
client = ModelClient()
# Test texts
texts = [
"Example text in English",
"Exemple de texte en français",
"مثال نص باللغة العربية"
]
try:
# Get model status
status = client.get_model_status()
print("Model Status:", json.dumps(status, indent=2))
# Make predictions
predictions = client.predict(texts)
print("\nPredictions:", json.dumps(predictions, indent=2))
except Exception as e:
print(f"Error: {e}")
```

---------------------------------------------------------------------------

## File: s3Handler.py
- Path: `/root/git/text-meaning/Tache7/scripts/utils/s3Handler.py`
- Size: 2.20 KB
- Extension: .py
- Lines of code: 53

```py
import boto3
import os
from botocore.exceptions import ClientError
class S3Handler:
def __init__(self):
self.s3_client = boto3.client(
's3',
aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
region_name=os.getenv('AWS_REGION')
)
self.raw_bucket = os.getenv('S3_RAW_BUCKET')
self.processed_bucket = os.getenv('S3_PROCESSED_BUCKET')
self.models_bucket = os.getenv('S3_MODELS_BUCKET')
def upload_file(self, file_path: str, bucket: str, object_name: str = None) -> bool:
"""Upload a file to S3 bucket"""
if object_name is None:
object_name = os.path.basename(file_path)
try:
self.s3_client.upload_file(file_path, bucket, object_name)
return True
except ClientError as e:
print(f"Error uploading file to S3: {e}")
return False
def download_file(self, bucket: str, object_name: str, file_path: str) -> bool:
"""Download a file from S3 bucket"""
try:
self.s3_client.download_file(bucket, object_name, file_path)
return True
except ClientError as e:
print(f"Error downloading file from S3: {e}")
return False
def list_files(self, bucket: str, prefix: str = '') -> list:
"""List files in S3 bucket with optional prefix"""
try:
response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
return [obj['Key'] for obj in response.get('Contents', [])]
except ClientError as e:
print(f"Error listing files in S3: {e}")
return []
def save_text_data(self, text: str, filename: str, is_processed: bool = False) -> bool:
"""Save text data to appropriate bucket"""
bucket = self.processed_bucket if is_processed else self.raw_bucket
try:
self.s3_client.put_object(
Bucket=bucket,
Key=filename,
Body=text.encode('utf-8')
)
return True
except ClientError as e:
print(f"Error saving text data to S3: {e}")
return False
```

---------------------------------------------------------------------------

## File: dashboards.yml
- Path: `/root/git/text-meaning/Tache7/grafana/provisioning/dashboards/dashboards.yml`
- Size: 187.00 B
- Extension: .yml
- Lines of code: 10

```yml
apiVersion: 1
providers:
- name: 'Default'
orgId: 1
folder: ''
type: file
disableDeletion: false
editable: true
options:
path: /var/lib/grafana/dashboards
```

---------------------------------------------------------------------------

## File: datasource.yml
- Path: `/root/git/text-meaning/Tache7/grafana/provisioning/datasources/datasource.yml`
- Size: 159.00 B
- Extension: .yml
- Lines of code: 8

```yml
apiVersion: 1
datasources:
- name: Prometheus
type: prometheus
access: proxy
url: http://prometheus:9090
isDefault: true
editable: false
```

---------------------------------------------------------------------------

## File: text_processor.py
- Path: `/root/git/text-meaning/Tache7/scripts/spark/jobs/text_processor.py`
- Size: 3.25 KB
- Extension: .py
- Lines of code: 76

```py
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, to_json, struct
from pyspark.sql.types import StructType, StructField, StringType, MapType, TimestampType
from datetime import datetime
import os
class TextProcessorS3:
def __init__(self, spark):
self.spark = spark
self.s3_raw_bucket = os.getenv('S3_RAW_BUCKET')
self.s3_processed_bucket = os.getenv('S3_PROCESSED_BUCKET')
def write_to_s3(self, batch_df, batch_id):
"""Write batch to S3 with partitioning"""
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# Convert DataFrame to format suitable for S3
output_df = batch_df.select(
col("language"),
to_json(struct("*")).alias("data")
)
# Write to S3 with partitioning by language and timestamp
s3_path = f"s3a://{self.s3_processed_bucket}/processed/language={batch_df.language}/batch={timestamp}"
output_df.write \
.mode("append") \
.json(s3_path)
print(f"Batch {batch_id} written to {s3_path}")
def create_spark_session():
return SparkSession.builder \
.appName("TextProcessorS3") \
.config("spark.jars.packages",
"org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2,"
"org.apache.hadoop:hadoop-aws:3.3.2,"
"com.amazonaws:aws-java-sdk-bundle:1.11.1026") \
.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
.config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID")) \
.config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY")) \
.config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
.config("spark.hadoop.fs.s3a.path.style.access", "true") \
.config("spark.streaming.stopGracefullyOnShutdown", "true") \
.getOrCreate()
def create_streaming_query(spark, processor):
# Schema for Kafka messages
schema = StructType([
StructField("text", StringType(), True),
StructField("language", StringType(), True),
StructField("metadata", MapType(StringType(), StringType()), True),
StructField("timestamp", TimestampType(), True)
])
# Read from Kafka
stream_df = spark.readStream \
.format("kafka") \
.option("kafka.bootstrap.servers", "kafka:29092") \
.option("subscribe", "raw-text-fr,raw-text-en,raw-text-ar") \
.load()
# Parse JSON and process
parsed_df = stream_df.select(
from_json(col("value").cast("string"), schema).alias("parsed")
).select("parsed.*")
# Write to both Kafka and S3
query = parsed_df.writeStream \
.foreachBatch(processor.write_to_s3) \
.outputMode("append") \
.option("checkpointLocation", "/tmp/checkpoint/s3") \
.trigger(processingTime="1 minute") \
.start()
return query
def main():
spark = create_spark_session()
processor = TextProcessorS3(spark)
query = create_streaming_query(spark, processor)
try:
query.awaitTermination()
except KeyboardInterrupt:
print("Stopping the streaming query...")
query.stop()
spark.stop()
if __name__ == "__main__":
main()
```

---------------------------------------------------------------------------