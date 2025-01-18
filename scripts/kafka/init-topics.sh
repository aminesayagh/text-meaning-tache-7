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