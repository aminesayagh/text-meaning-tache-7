# Project Tree Structure
```plaintext
.
|-- config
|   |-- kafka
|   `-- spark
|-- data
|   |-- input
|   `-- output
|-- doc
|   `-- cw.sh
|-- docs
|   |-- doc.md
|   `-- doc-project-tree.md
|-- grafana
|   `-- provisioning
|       |-- dashboards
|       |   `-- dashboards.yml
|       `-- datasources
|           `-- datasource.yml
|-- logs
|-- models
|   |-- models.config
|   `-- monitoring.config
|-- prometheus
|   `-- prometheus.yml
|-- scripts
|   |-- kafka
|   |   |-- init-topics.sh
|   |   `-- text_producer.py
|   |-- spark
|   |   |-- jobs
|   |   |   `-- text_processor.py
|   |   `-- submit-job.sh
|   `-- utils
|       |-- model_client.py
|       `-- s3Handler.py
|-- docker-compose.yml
|-- .env
|-- nginx.conf
|-- requirements.txt
`-- spark-defaults.conf

20 directories, 19 files
```
