package com.cryptoanom.features;

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
                avg("price").alias("rolling_price_mean"),
                coalesce(stddev("price"), lit(0.0)).alias("rolling_price_std"),
                avg("volume").alias("rolling_volume_mean"),
                coalesce(stddev("volume"), lit(0.0)).alias("rolling_volume_std"),
                avg("log_return").alias("log_return"),
                max("volume").alias("volume"),
                max(col("is_anomaly").cast("integer")).alias("is_anomaly")
        );
    }
}
