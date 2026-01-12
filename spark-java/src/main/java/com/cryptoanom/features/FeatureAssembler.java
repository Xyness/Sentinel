package com.cryptoanom.features;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class FeatureAssembler {

    public static Dataset<Row> buildFeatures(Dataset<Row> stream) {

        Dataset<Row> rolling = RollingFeatures.addRollingStats(stream, "1 minute");

        Dataset<Row> enriched = TechnicalIndicators.addZScore(rolling);

        return enriched;
    }
}
