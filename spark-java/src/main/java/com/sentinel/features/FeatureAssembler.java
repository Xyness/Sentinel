package com.sentinel.features;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.lit;

/**
 * What leaves the streaming job: one row per symbol per window.
 */
public class FeatureAssembler {

    /** The contract with preprocess.py, the scorer and the API schema. */
    public static final String[] FEATURE_COLUMNS = {
            "abs_return_max",
            "return_std",
            "price_range_rel",
            "volume_max_ratio",
            "volume_cv"
    };

    public static Dataset<Row> buildFeatures(Dataset<Row> stream,
                                             String windowDuration,
                                             int minEvents) {

        Dataset<Row> aggregates = RollingFeatures.aggregate(stream, windowDuration);
        Dataset<Row> withRatios = TechnicalIndicators.addRatios(aggregates);

        return withRatios
                // The first and last windows of a run are partial, and a window
                // holding two events describes the sampling rather than the
                // market. Dropping them costs a row a restart and keeps the
                // training set from being seeded with noise.
                .filter(col("event_count").geq(lit(minEvents)))
                .select(
                        col("symbol"),
                        col("window").getField("start").alias("window_start"),
                        col("event_count"),
                        col("abs_return_max"),
                        col("return_std"),
                        col("price_range_rel"),
                        col("volume_max_ratio"),
                        col("volume_cv"),
                        col("is_anomaly")
                );
    }
}
