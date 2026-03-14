package com.cryptoanom.streaming;

import com.cryptoanom.features.FeatureAssembler;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import static org.apache.spark.sql.functions.*;

public class CryptoStreamJob {

    private static final String DEFAULT_KAFKA_SERVERS = "localhost:9092";
    private static final String DEFAULT_KAFKA_TOPIC = "crypto-market";
    private static final String DEFAULT_OUTPUT_PATH = "data/features";
    private static final String DEFAULT_CHECKPOINT_PATH = "data/checkpoints/cryptoanom";

    public static void main(String[] args) throws Exception {

        String kafkaServers = System.getenv().getOrDefault("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_KAFKA_SERVERS);
        String kafkaTopic = System.getenv().getOrDefault("KAFKA_TOPIC", DEFAULT_KAFKA_TOPIC);
        String outputPath = System.getenv().getOrDefault("OUTPUT_PATH", DEFAULT_OUTPUT_PATH);
        String checkpointPath = System.getenv().getOrDefault("CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH);

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
                .option("kafka.bootstrap.servers", kafkaServers)
                .option("subscribe", kafkaTopic)
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
                .option("path", outputPath)
                .option("checkpointLocation", checkpointPath)
                .partitionBy("symbol")
                .outputMode("append")
                .start();

        query.awaitTermination();
    }
}
