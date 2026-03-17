package com.sentinel.features;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.col;

public class FeatureAssembler {

    public static Dataset<Row> buildFeatures(Dataset<Row> stream) {

        Dataset<Row> rolling = RollingFeatures.addRollingStats(stream, "1 minute");

        Dataset<Row> enriched = TechnicalIndicators.addZScore(rolling);

        // Select only the columns needed downstream (ML training + API)
        return enriched.select(
                col("symbol"),
                col("z_score_price"),
                col("z_score_log_return"),
                col("z_score_volume"),
                col("rolling_price_std"),
                col("rolling_volume_std"),
                col("is_anomaly")
        );
    }
}
