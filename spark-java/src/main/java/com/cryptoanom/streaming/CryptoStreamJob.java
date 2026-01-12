package com.cryptoanom.streaming;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import static org.apache.spark.sql.functions.*;

public class CryptoStreamJob {

    public static void main(String[] args) throws Exception {

        SparkSession spark = SparkSession.builder()
                .appName("CryptoAnom-Streaming")
                .master("local[*]")
                .getOrCreate();

        spark.sparkContext().setLogLevel("WARN");

        StructType schema = new StructType()
                .add("timestamp", DataTypes.LongType)
                .add("symbol", DataTypes.StringType)
                .add("price", DataTypes.DoubleType)
                .add("volume", DataTypes.DoubleType)
                .add("log_return", DataTypes.DoubleType)
                .add("is_anomaly", DataTypes.BooleanType)
                .add("anomaly_type", DataTypes.StringType);

        Dataset<Row> rawKafka = spark.readStream()
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("subscribe", "crypto-market")
                .option("startingOffsets", "latest")
                .load();

        Dataset<Row> marketStream = rawKafka
                .selectExpr("CAST(value AS STRING)")
                .select(from_json(col("value"), schema).as("data"))
                .select("data.*")
                .withColumn("event_time",
                        to_timestamp(from_unixtime(col("timestamp"))));

        Dataset<Row> withWatermark = marketStream
                .withWatermark("event_time", "1 minute");

        Dataset<Row> featureStream = FeatureAssembler.buildFeatures(withWatermark);

        StreamingQuery query = featureStream.writeStream()
                .format("parquet")
                .option("path", "data/features")
                .option("checkpointLocation", "data/checkpoints/cryptoanom")
                .partitionBy("symbol")
                .outputMode("append")
                .start();

        query.awaitTermination();
    }
}
