package com.cryptoanom.features;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.*;

public class TechnicalIndicators {

    public static Dataset<Row> addZScore(Dataset<Row> df) {

        return df.withColumn(
                "z_score_price",
                col("log_return").divide(col("rolling_price_std"))
        ).withColumn(
                "z_score_volume",
                col("volume").minus(col("rolling_volume_mean"))
                        .divide(col("rolling_volume_std"))
        );
    }
}
