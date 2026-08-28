package com.sentinel.features;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.*;

/**
 * Per-symbol aggregates over one time window.
 *
 * Everything here is order independent on purpose. An aggregation shuffles, so
 * `last()` returns whichever row the executor happened to see last rather than
 * the last one in time, and a feature built on it is not reproducible. Sums,
 * extrema and deviations do not care what order they arrive in.
 *
 * These are raw quantities in the units of the market: prices, volumes, log
 * returns. TechnicalIndicators turns them into the dimensionless ratios the
 * model is fitted on.
 */
public class RollingFeatures {

    public static Dataset<Row> aggregate(Dataset<Row> df, String windowDuration) {

        return df.groupBy(
                col("symbol"),
                window(col("event_time"), windowDuration)
        ).agg(
                count(lit(1)).alias("event_count"),

                avg("price").alias("price_mean"),
                min("price").alias("price_min"),
                max("price").alias("price_max"),

                // The single most violent tick of the window. This is the one
                // that has to survive: an anomaly injected in the middle of a
                // window is invisible to anything that only looks at the edges.
                max(abs(col("log_return"))).alias("abs_log_return_max"),
                coalesce(stddev("log_return"), lit(0.0)).alias("log_return_std"),

                avg("volume").alias("volume_mean"),
                max("volume").alias("volume_max"),
                coalesce(stddev("volume"), lit(0.0)).alias("volume_std"),

                // The label covers the whole window, so the features have to as
                // well, or the model is asked to predict an event it cannot see.
                max(col("is_anomaly").cast("integer")).alias("is_anomaly")
        );
    }
}
