package com.cryptosentinel.features;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.*;

public class TechnicalIndicators {

    public static Dataset<Row> addZScore(Dataset<Row> df) {

        return df.withColumn(
                "z_score_price",
                when(col("rolling_price_std").equalTo(0), lit(0.0))
                        .otherwise(
                                col("price").minus(col("rolling_price_mean"))
                                        .divide(col("rolling_price_std"))
                        )
        ).withColumn(
                "z_score_log_return",
                when(col("rolling_log_return_std").equalTo(0), lit(0.0))
                        .otherwise(
                                col("log_return").minus(col("rolling_log_return_mean"))
                                        .divide(col("rolling_log_return_std"))
                        )
        ).withColumn(
                "z_score_volume",
                when(col("rolling_volume_std").equalTo(0), lit(0.0))
                        .otherwise(
                                col("volume").minus(col("rolling_volume_mean"))
                                        .divide(col("rolling_volume_std"))
                        )
        );
    }
}
