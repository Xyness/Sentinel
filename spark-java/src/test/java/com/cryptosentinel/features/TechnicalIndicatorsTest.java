package com.cryptosentinel.features;

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

    private Dataset<Row> createTestDF(double price, double rollingPriceMean,
                                       double rollingPriceStd, double logReturn,
                                       double rollingLogReturnMean, double rollingLogReturnStd,
                                       double volume, double rollingVolumeMean,
                                       double rollingVolumeStd) {
        StructType schema = new StructType()
                .add("price", DataTypes.DoubleType)
                .add("rolling_price_mean", DataTypes.DoubleType)
                .add("rolling_price_std", DataTypes.DoubleType)
                .add("log_return", DataTypes.DoubleType)
                .add("rolling_log_return_mean", DataTypes.DoubleType)
                .add("rolling_log_return_std", DataTypes.DoubleType)
                .add("volume", DataTypes.DoubleType)
                .add("rolling_volume_mean", DataTypes.DoubleType)
                .add("rolling_volume_std", DataTypes.DoubleType);

        List<Row> data = Arrays.asList(
                RowFactory.create(price, rollingPriceMean, rollingPriceStd,
                        logReturn, rollingLogReturnMean, rollingLogReturnStd,
                        volume, rollingVolumeMean, rollingVolumeStd)
        );

        return spark.createDataFrame(data, schema);
    }

    @Test
    public void testZScorePriceCalculation() {
        // price=100, mean=90, std=5 → z = (100-90)/5 = 2.0
        Dataset<Row> df = createTestDF(100.0, 90.0, 5.0, 0.05, 0.02, 0.01, 100.0, 80.0, 20.0);
        Dataset<Row> result = TechnicalIndicators.addZScore(df);
        Row row = result.first();

        double zScorePrice = row.getAs("z_score_price");
        assertEquals(2.0, zScorePrice, 1e-6);
    }

    @Test
    public void testZScoreLogReturnCalculation() {
        // logReturn=0.05, mean=0.02, std=0.01 → z = (0.05-0.02)/0.01 = 3.0
        Dataset<Row> df = createTestDF(100.0, 90.0, 5.0, 0.05, 0.02, 0.01, 100.0, 80.0, 20.0);
        Dataset<Row> result = TechnicalIndicators.addZScore(df);
        Row row = result.first();

        double zScoreLogReturn = row.getAs("z_score_log_return");
        assertEquals(3.0, zScoreLogReturn, 1e-6);
    }

    @Test
    public void testZScoreVolumeCalculation() {
        // volume=100, mean=80, std=20 → z = (100-80)/20 = 1.0
        Dataset<Row> df = createTestDF(100.0, 90.0, 5.0, 0.05, 0.02, 0.01, 100.0, 80.0, 20.0);
        Dataset<Row> result = TechnicalIndicators.addZScore(df);
        Row row = result.first();

        double zScoreVolume = row.getAs("z_score_volume");
        assertEquals(1.0, zScoreVolume, 1e-6);
    }

    @Test
    public void testZScoreWithZeroStd() {
        Dataset<Row> df = createTestDF(100.0, 90.0, 0.0, 0.05, 0.02, 0.0, 100.0, 80.0, 0.0);
        Dataset<Row> result = TechnicalIndicators.addZScore(df);
        Row row = result.first();

        // All three z-scores should return 0.0 when std is 0
        double zScorePrice = row.getAs("z_score_price");
        double zScoreLogReturn = row.getAs("z_score_log_return");
        double zScoreVolume = row.getAs("z_score_volume");
        assertEquals(0.0, zScorePrice, 1e-6);
        assertEquals(0.0, zScoreLogReturn, 1e-6);
        assertEquals(0.0, zScoreVolume, 1e-6);
    }

    @Test
    public void testNegativeZScores() {
        // price=80, mean=90, std=5 → z = (80-90)/5 = -2.0
        // logReturn=-0.03, mean=0.02, std=0.01 → z = (-0.03-0.02)/0.01 = -5.0
        // volume=50, mean=80, std=20 → z = (50-80)/20 = -1.5
        Dataset<Row> df = createTestDF(80.0, 90.0, 5.0, -0.03, 0.02, 0.01, 50.0, 80.0, 20.0);
        Dataset<Row> result = TechnicalIndicators.addZScore(df);
        Row row = result.first();

        double zScorePrice = row.getAs("z_score_price");
        assertEquals(-2.0, zScorePrice, 1e-6);

        double zScoreLogReturn = row.getAs("z_score_log_return");
        assertEquals(-5.0, zScoreLogReturn, 1e-6);

        double zScoreVolume = row.getAs("z_score_volume");
        assertEquals(-1.5, zScoreVolume, 1e-6);
    }
}
