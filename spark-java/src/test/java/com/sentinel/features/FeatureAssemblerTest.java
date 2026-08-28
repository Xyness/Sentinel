package com.sentinel.features;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import org.junit.jupiter.api.*;

import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The whole window, end to end, over a batch DataFrame.
 *
 * These are the tests that pin down the bug this feature set exists to fix. The
 * old features z-scored the last price of a window against that same window's
 * own mean and deviation, which made the score of a step depend only on where
 * in the window it landed and not at all on how big it was, and left an
 * anomaly in the middle of a window invisible.
 */
public class FeatureAssemblerTest {

    private static SparkSession spark;

    /** 2026-01-01T00:00:00Z, a whole number of minutes since the epoch. */
    private static final long BASE_MILLIS = 1767225600000L;

    @BeforeAll
    public static void setup() {
        spark = SparkSession.builder()
                .appName("FeatureAssemblerTest")
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

    // -- building a window of events ------------------------------------------

    private static final StructType EVENT_SCHEMA = new StructType()
            .add("event_time", DataTypes.TimestampType)
            .add("symbol", DataTypes.StringType)
            .add("price", DataTypes.DoubleType)
            .add("volume", DataTypes.DoubleType)
            .add("log_return", DataTypes.DoubleType)
            .add("is_anomaly", DataTypes.BooleanType);

    /** A price path that steps by `factor` at `jumpAt`, and stays there. */
    private double[] pricesWithJump(int n, double base, int jumpAt, double factor) {
        double[] prices = new double[n];
        double price = base;
        for (int i = 0; i < n; i++) {
            if (i == jumpAt) {
                price *= factor;
            }
            prices[i] = price;
        }
        return prices;
    }

    private double[] flat(int n, double value) {
        double[] values = new double[n];
        java.util.Arrays.fill(values, value);
        return values;
    }

    private List<Row> events(String symbol, double[] prices, double[] volumes, int anomalyAt) {
        List<Row> rows = new ArrayList<>();
        for (int i = 0; i < prices.length; i++) {
            double logReturn = i == 0 ? 0.0 : Math.log(prices[i] / prices[i - 1]);
            rows.add(RowFactory.create(
                    new Timestamp(BASE_MILLIS + i * 1000L),
                    symbol,
                    prices[i],
                    volumes[i],
                    logReturn,
                    i == anomalyAt
            ));
        }
        return rows;
    }

    private Row featuresOf(List<Row> events, int minEvents) {
        Dataset<Row> stream = spark.createDataFrame(events, EVENT_SCHEMA);
        List<Row> out = FeatureAssembler.buildFeatures(stream, "1 minute", minEvents)
                .collectAsList();
        assertEquals(1, out.size(), "expected exactly one window");
        return out.get(0);
    }

    private Row featuresOf(List<Row> events) {
        return featuresOf(events, 1);
    }

    // -- the regression -------------------------------------------------------

    @Test
    public void testAJumpIsSeenWhereverInTheWindowItLands() {
        // The old z-score came out as sqrt(k/(n-k)): the same move scored 0.5
        // early in the window and 7.7 at the end of it. Position is not a
        // property of the anomaly and has no business in the feature.
        int n = 30;
        Row early = featuresOf(events("BTC-USDT",
                pricesWithJump(n, 100.0, 5, 1.10), flat(n, 10.0), 5));
        Row late = featuresOf(events("BTC-USDT",
                pricesWithJump(n, 100.0, 25, 1.10), flat(n, 10.0), 25));

        assertEquals(early.<Double>getAs("abs_return_max"),
                late.<Double>getAs("abs_return_max"), 1e-12);
        assertEquals(Math.log(1.10), early.<Double>getAs("abs_return_max"), 1e-12);
    }

    @Test
    public void testAJumpCarriesItsSizeIntoTheFeature() {
        // The other half of the same bug: J cancelled out of the old z-score,
        // so a 15 % crash and a 0.3 % drift scored identically.
        int n = 30;
        Row small = featuresOf(events("BTC-USDT",
                pricesWithJump(n, 100.0, 15, 1.003), flat(n, 10.0), 15));
        Row large = featuresOf(events("BTC-USDT",
                pricesWithJump(n, 100.0, 15, 1.15), flat(n, 10.0), 15));

        assertEquals(Math.log(1.003), small.<Double>getAs("abs_return_max"), 1e-12);
        assertEquals(Math.log(1.15), large.<Double>getAs("abs_return_max"), 1e-12);
        assertTrue(large.<Double>getAs("abs_return_max")
                        > small.<Double>getAs("abs_return_max") * 10,
                "a large move has to score well above a small one");
    }

    @Test
    public void testAQuietWindowSitsAtTheFloor() {
        int n = 30;
        Row row = featuresOf(events("BTC-USDT", flat(n, 100.0), flat(n, 10.0), -1));

        assertEquals(0.0, row.<Double>getAs("abs_return_max"), 1e-12);
        assertEquals(0.0, row.<Double>getAs("return_std"), 1e-12);
        assertEquals(0.0, row.<Double>getAs("price_range_rel"), 1e-12);
        assertEquals(1.0, row.<Double>getAs("volume_max_ratio"), 1e-12);
        assertEquals(0.0, row.<Double>getAs("volume_cv"), 1e-12);
        assertEquals(0, row.<Integer>getAs("is_anomaly").intValue());
    }

    @Test
    public void testAQuietWindowAndAJumpedOneAreNotTheSameRow() {
        int n = 30;
        Row quiet = featuresOf(events("BTC-USDT", flat(n, 100.0), flat(n, 10.0), -1));
        Row jumped = featuresOf(events("BTC-USDT",
                pricesWithJump(n, 100.0, 15, 1.10), flat(n, 10.0), 15));

        assertTrue(jumped.<Double>getAs("abs_return_max") > quiet.<Double>getAs("abs_return_max"));
        assertTrue(jumped.<Double>getAs("price_range_rel") > quiet.<Double>getAs("price_range_rel"));
    }

    // -- order independence ---------------------------------------------------

    @Test
    public void testShufflingTheEventsChangesNothing() {
        // An aggregation shuffles, so `last()` used to return whichever row an
        // executor saw last. Every feature here is an extremum, a sum or a
        // deviation, and none of them can tell.
        int n = 30;
        List<Row> ordered = events("BTC-USDT",
                pricesWithJump(n, 100.0, 12, 1.08), flat(n, 10.0), 12);
        List<Row> shuffled = new ArrayList<>(ordered);
        Collections.shuffle(shuffled, new java.util.Random(7));

        Row a = featuresOf(ordered);
        Row b = featuresOf(shuffled);

        for (String feature : FeatureAssembler.FEATURE_COLUMNS) {
            assertEquals(a.<Double>getAs(feature), b.<Double>getAs(feature), 1e-12, feature);
        }
    }

    // -- volume ---------------------------------------------------------------

    @Test
    public void testAVolumeSpikeShowsUpWithoutTouchingThePriceFeatures() {
        int n = 30;
        double[] volumes = flat(n, 10.0);
        volumes[8] = 80.0;

        Row row = featuresOf(events("BTC-USDT", flat(n, 100.0), volumes, 8));

        assertEquals(0.0, row.<Double>getAs("abs_return_max"), 1e-12);
        assertTrue(row.<Double>getAs("volume_max_ratio") > 6.0,
                "an eightfold trade should be well clear of the window average");
        assertTrue(row.<Double>getAs("volume_cv") > 0.5);
    }

    // -- the label ------------------------------------------------------------

    @Test
    public void testOneFlaggedEventMarksTheWholeWindow() {
        // The label is what the features have to be able to explain. It covers
        // the window, so the features do too.
        int n = 30;
        Row row = featuresOf(events("BTC-USDT",
                pricesWithJump(n, 100.0, 14, 1.10), flat(n, 10.0), 14));
        assertEquals(1, row.<Integer>getAs("is_anomaly").intValue());
    }

    // -- partial windows ------------------------------------------------------

    @Test
    public void testWindowsWithTooFewEventsAreDropped() {
        List<Row> thin = events("BTC-USDT", flat(4, 100.0), flat(4, 10.0), -1);
        Dataset<Row> stream = spark.createDataFrame(thin, EVENT_SCHEMA);

        assertTrue(FeatureAssembler.buildFeatures(stream, "1 minute", 10)
                .collectAsList().isEmpty());
        assertEquals(1, FeatureAssembler.buildFeatures(stream, "1 minute", 4)
                .collectAsList().size());
    }

    @Test
    public void testTheRowCarriesItsWindowAndItsCount() {
        int n = 30;
        Row row = featuresOf(events("BTC-USDT", flat(n, 100.0), flat(n, 10.0), -1));

        assertEquals("BTC-USDT", row.<String>getAs("symbol"));
        assertEquals(30L, row.<Long>getAs("event_count").longValue());
        assertEquals(BASE_MILLIS, row.<Timestamp>getAs("window_start").getTime());
    }
}
