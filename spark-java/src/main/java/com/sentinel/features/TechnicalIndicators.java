package com.sentinel.features;

import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import static org.apache.spark.sql.functions.*;

/**
 * The five features the model is fitted on, all of them dimensionless.
 *
 * They are ratios rather than z-scores against the window's own mean and
 * deviation. A z-score computed that way is worthless here: for a step of size
 * J landing at position k of an n-event window, the deviation from the mean is
 * J*k/n and the standard deviation is J*sqrt(k*(n-k))/n, so the z-score comes
 * out as sqrt(k/(n-k)) and J cancels. A 15 % flash crash and a 0.3 % drift
 * score identically, and both sit around 1.0, which is where an ordinary quiet
 * window sits too. The anomaly inflates the very deviation used to normalise
 * it and hides itself.
 *
 * A ratio against a quantity the anomaly does not move keeps the magnitude.
 * Being dimensionless also makes the pairs comparable: an absolute price
 * deviation puts BTC two orders of magnitude away from BNB before anything
 * unusual has happened, and the model spends its budget separating symbols
 * rather than separating anomalies.
 */
public class TechnicalIndicators {

    public static Dataset<Row> addRatios(Dataset<Row> df) {

        return df
                // The largest absolute log return in the window. Already
                // dimensionless, already a magnitude: log(1.10) = 0.0953 for a
                // ten percent jump wherever in the window it lands.
                .withColumn("abs_return_max",
                        coalesce(col("abs_log_return_max"), lit(0.0)))

                // Realised volatility over the window.
                .withColumn("return_std",
                        coalesce(col("log_return_std"), lit(0.0)))

                // How far the price travelled, as a fraction of its own level.
                .withColumn("price_range_rel",
                        ratio(col("price_max").minus(col("price_min")), col("price_mean")))

                // The biggest single trade against the window's typical one.
                // Sits at 1.0 on a flat window and climbs with a volume spike.
                .withColumn("volume_max_ratio",
                        ratio(col("volume_max"), col("volume_mean")))

                // Volume dispersion, normalised so it means the same thing on
                // a pair that trades in thousands and one that trades in tens.
                .withColumn("volume_cv",
                        ratio(col("volume_std"), col("volume_mean")));
    }

    /**
     * A division that returns 0 rather than a null or an infinity.
     *
     * A window with one event has no deviation and a window of free coins has
     * no price. Neither is an anomaly, and a NaN in one feature propagates
     * through the scaler and out the far side of training.
     */
    private static Column ratio(Column numerator, Column denominator) {
        return when(denominator.isNull().or(denominator.equalTo(0)), lit(0.0))
                .otherwise(numerator.divide(denominator));
    }
}
