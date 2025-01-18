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