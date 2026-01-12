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
                stddev("price").alias("rolling_price_std"),
                avg("volume").alias("rolling_volume_mean"),
                stddev("volume").alias("rolling_volume_std")
        );
    }
}
