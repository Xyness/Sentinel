package com.cryptoanom.features;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import org.junit.jupiter.api.*;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class TechnicalIndicatorsTest {

    private static SparkSession spark;

    @BeforeAll
    public static void setup() {
        spark = SparkSession.builder()
                .appName("TechnicalIndicatorsTest")
                .master("local[*]")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
    }

    @AfterAll
    public static void teardown() {
        if (spark != null) {
            spark.stop();
        }
    }

    private Dataset<Row> createTestDF(double logReturn, double rollingPriceMean,
                                       double rollingPriceStd, double volume,
                                       double rollingVolumeMean, double rollingVolumeStd) {
        StructType schema = new StructType()
                .add("log_return", DataTypes.DoubleType)
                .add("rolling_price_mean", DataTypes.DoubleType)
                .add("rolling_price_std", DataTypes.DoubleType)
                .add("volume", DataTypes.DoubleType)
                .add("rolling_volume_mean", DataTypes.DoubleType)
                .add("rolling_volume_std", DataTypes.DoubleType);

        List<Row> data = Arrays.asList(
                RowFactory.create(logReturn, rollingPriceMean, rollingPriceStd,
                        volume, rollingVolumeMean, rollingVolumeStd)
        );

        return spark.createDataFrame(data, schema);
    }

    @Test
    public void testZScorePriceCalculation() {
        Dataset<Row> df = createTestDF(0.05, 0.02, 0.01, 100.0, 80.0, 20.0);
        Dataset<Row> result = TechnicalIndicators.addZScore(df);
        Row row = result.first();

        // z_score_price = (log_return - rolling_price_mean) / rolling_price_std
        // = (0.05 - 0.02) / 0.01 = 3.0
        double zScorePrice = row.getAs("z_score_price");
        assertEquals(3.0, zScorePrice, 1e-6);
    }

    @Test
    public void testZScoreVolumeCalculation() {
        Dataset<Row> df = createTestDF(0.05, 0.02, 0.01, 100.0, 80.0, 20.0);
        Dataset<Row> result = TechnicalIndicators.addZScore(df);
        Row row = result.first();

        // z_score_volume = (volume - rolling_volume_mean) / rolling_volume_std
        // = (100 - 80) / 20 = 1.0
        double zScoreVolume = row.getAs("z_score_volume");
        assertEquals(1.0, zScoreVolume, 1e-6);
    }

    @Test
    public void testZScoreWithZeroStd() {
        Dataset<Row> df = createTestDF(0.05, 0.02, 0.0, 100.0, 80.0, 0.0);
        Dataset<Row> result = TechnicalIndicators.addZScore(df);
        Row row = result.first();

        // Should return 0.0 when std is 0 (division by zero protection)
        double zScorePrice = row.getAs("z_score_price");
        double zScoreVolume = row.getAs("z_score_volume");
        assertEquals(0.0, zScorePrice, 1e-6);
        assertEquals(0.0, zScoreVolume, 1e-6);
    }

    @Test
    public void testNegativeZScores() {
        Dataset<Row> df = createTestDF(-0.03, 0.02, 0.01, 50.0, 80.0, 20.0);
        Dataset<Row> result = TechnicalIndicators.addZScore(df);
        Row row = result.first();

        // z_score_price = (-0.03 - 0.02) / 0.01 = -5.0
        double zScorePrice = row.getAs("z_score_price");
        assertEquals(-5.0, zScorePrice, 1e-6);

        // z_score_volume = (50 - 80) / 20 = -1.5
        double zScoreVolume = row.getAs("z_score_volume");
        assertEquals(-1.5, zScoreVolume, 1e-6);
    }
}
