package com.sentinel.features;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import org.junit.jupiter.api.*;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** The ratio arithmetic, one hand-built aggregate row at a time. */
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

    private Row ratiosOf(double absLogReturnMax, double logReturnStd,
                         double priceMin, double priceMax, double priceMean,
                         double volumeMax, double volumeMean, double volumeStd) {

        StructType schema = new StructType()
                .add("abs_log_return_max", DataTypes.DoubleType)
                .add("log_return_std", DataTypes.DoubleType)
                .add("price_min", DataTypes.DoubleType)
                .add("price_max", DataTypes.DoubleType)
                .add("price_mean", DataTypes.DoubleType)
                .add("volume_max", DataTypes.DoubleType)
                .add("volume_mean", DataTypes.DoubleType)
                .add("volume_std", DataTypes.DoubleType);

        List<Row> data = Arrays.asList(
                RowFactory.create(absLogReturnMax, logReturnStd, priceMin, priceMax,
                        priceMean, volumeMax, volumeMean, volumeStd)
        );

        return TechnicalIndicators.addRatios(spark.createDataFrame(data, schema)).first();
    }

    @Test
    public void testPriceRangeIsAFractionOfTheLevel() {
        // range 10 over a mean of 100 -> 0.10
        Row row = ratiosOf(0.05, 0.01, 95.0, 105.0, 100.0, 30.0, 10.0, 4.0);
        assertEquals(0.10, row.<Double>getAs("price_range_rel"), 1e-9);
    }

    @Test
    public void testTheSameMoveOnAnyPairGivesTheSameNumber() {
        // A one percent range, once around 100 and once around 43000. An
        // absolute deviation would put these two orders of magnitude apart and
        // the model would spend itself telling BTC from BNB.
        Row cheap = ratiosOf(0.01, 0.002, 99.5, 100.5, 100.0, 12.0, 10.0, 1.0);
        Row dear = ratiosOf(0.01, 0.002, 42785.0, 43215.0, 43000.0, 12.0, 10.0, 1.0);

        assertEquals(cheap.<Double>getAs("price_range_rel"),
                dear.<Double>getAs("price_range_rel"), 1e-9);
    }

    @Test
    public void testVolumeRatioSitsAtOneOnAFlatWindow() {
        Row row = ratiosOf(0.0, 0.0, 100.0, 100.0, 100.0, 10.0, 10.0, 0.0);
        assertEquals(1.0, row.<Double>getAs("volume_max_ratio"), 1e-9);
        assertEquals(0.0, row.<Double>getAs("volume_cv"), 1e-9);
    }

    @Test
    public void testVolumeRatioClimbsWithASpike() {
        // one trade eight times the window's average
        Row row = ratiosOf(0.0, 0.0, 100.0, 100.0, 100.0, 80.0, 10.0, 22.0);
        assertEquals(8.0, row.<Double>getAs("volume_max_ratio"), 1e-9);
        assertEquals(2.2, row.<Double>getAs("volume_cv"), 1e-9);
    }

    @Test
    public void testTheReturnFeaturesPassStraightThrough() {
        Row row = ratiosOf(0.0953, 0.0184, 95.0, 105.0, 100.0, 30.0, 10.0, 4.0);
        assertEquals(0.0953, row.<Double>getAs("abs_return_max"), 1e-9);
        assertEquals(0.0184, row.<Double>getAs("return_std"), 1e-9);
    }

    @Test
    public void testAZeroDenominatorGivesZeroRatherThanANaN() {
        // A window with no price and no volume is not an anomaly, and a NaN
        // here propagates through the scaler and out the far side of training.
        Row row = ratiosOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        assertEquals(0.0, row.<Double>getAs("price_range_rel"), 1e-9);
        assertEquals(0.0, row.<Double>getAs("volume_max_ratio"), 1e-9);
        assertEquals(0.0, row.<Double>getAs("volume_cv"), 1e-9);

        for (String feature : FeatureAssembler.FEATURE_COLUMNS) {
            double value = row.<Double>getAs(feature);
            assertFalse(Double.isNaN(value), feature + " came out NaN");
            assertFalse(Double.isInfinite(value), feature + " came out infinite");
        }
    }
}
