package com.cryptosentinel.features;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.*;

public class RollingFeatures {

    public static Dataset<Row> addRollingStats(
            Dataset<Row> df,
            String windowDuration
    ) {

        return df.groupBy(
                col("symbol"),
                window(col("event_time"), windowDuration)
        ).agg(
                avg("log_return").alias("rolling_log_return_mean"),
                coalesce(stddev("log_return"), lit(0.0)).alias("rolling_log_return_std"),
                avg("volume").alias("rolling_volume_mean"),
                coalesce(stddev("volume"), lit(0.0)).alias("rolling_volume_std"),
                avg("price").alias("rolling_price_mean"),
                last("log_return").alias("log_return"),
                last("volume").alias("volume"),
                last("price").alias("price"),
                coalesce(stddev("price"), lit(0.0)).alias("rolling_price_std"),
                max(col("is_anomaly").cast("integer")).alias("is_anomaly")
        );
    }
}
