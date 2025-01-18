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