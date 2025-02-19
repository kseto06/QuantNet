#include "DataFramework.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>
#include <cmath>
#include <tuple>

namespace DataFramework {
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::vector<std::vector<std::vector<double>>> Tensor3D;

    Matrix parseData(const std::string& filename) {
        std::ifstream file(filename);
        Matrix data;
        std::string line;

        if (!file) {
            std::cerr << "Could not open file: " << filename
                  << " (" << std::strerror(errno) << ")" << std::endl;
            return data;
        }

        // Skip the heading at the top
        getline(file, line);

        // Parse the numerical data
        while (getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            std::vector<double> row;

            // Parse through the Date
            std::getline(ss, token, '-');
            row.push_back(std::stod(token)); //Year

            std::getline(ss, token, '-');
            row.push_back(std::stod(token)); //Month

            std::getline(ss, token, ',');
            row.push_back(std::stod(token)); //Day

            // Parse through all other columns
            while (getline(ss, token, ',')) {
                row.push_back(std::stod(token));
            }

            data.push_back(row);
        }
        file.close();
        return data;
    }

    //Function to convert a Date to a UnixTimestamp
    time_t UnixTimestamp(const double year, const double month, const double day) {
        struct std::tm time = {};
        time.tm_year = year - 1900;
        time.tm_mon = month - 1;
        time.tm_mday = day;
        time.tm_hour = 0;
        time.tm_min = 0;
        time.tm_sec = 0;

        return mktime(&time);
    }

    Matrix engineerData(const Matrix& data) {
        Matrix result(data.size(), std::vector<double>(16, 0.0)); // m x 16 features
        double EMA_12_day = 0.0;
        double EMA_26_day = 0.0;

        for (int row = 0; row < data.size(); row++) {
            /*
            * NOTE:
            * Row:
            *  0.    1.     2.   3.    4.    5.    6.     7
            * Year, Month, Day, Open, High, Low, Close, Volume
            *
            * Dropping these features due to a high p-value and/or low mutual information
            * Close % Change, Day, RSI, Daily Return, SMA + 2 STD, and SMA — 2 STD.
            *
            * Result:
            *   0.     1.       2.       3.           4.         5.          6.         7.         8.           9.           10.       11.        12.       13    14.  15
            *  Year, Month, Daily Var, Timestamp, 7-Day SMA, 7-Day STD, High-Close, Low-Open, Cumul Return, 14-Day EMA, Close Change, MACD, Stochastic Osc, ATR, ADX, DMI
            */

            //Construct features:
            double year = data[row][0];
            double month = data[row][1];
            double day = data[row][2]; //NOTE: Not used in result
            double daily_variation = data[row][4] - data[row][5]; //high - low
            double timestamp = static_cast<double>(UnixTimestamp(year, month, day) - UnixTimestamp(1970, 1, 1));

            // Populate time features to result
            result[row][0] = year;
            result[row][1] = month;
            result[row][2] = daily_variation;
            result[row][3] = timestamp;

            //Unused:
                // double index_hash
                // double daily_return = (row != 0) ? (data[row][6] - data[row-1][6])/data[row-1][6] : 0.0; // Daily Return: Percentage change of current and previous column's close

            double seven_day_SMA = 0.0; //Simple moving avg of Close column -- short-term trend of the index
            double seven_day_STD = 0.0; //Standard deviation of the Close column -- short-term variability of the index

            int availableDays = std::min(7, static_cast<int>(data.size()) - row);
            if (row > 7) {
                //Adding back with SMA
                for (int back = row; back > row - availableDays; back--) {
                    seven_day_SMA += data[back][6];
                }
                //Compute avg
                seven_day_SMA /= availableDays;

                //Standard deviation of the 7-day timeframe
                for (int back = row; back > row - availableDays; back--) {
                    seven_day_STD += std::pow(data[back][6] - seven_day_SMA, 2);
                }

                seven_day_STD = std::sqrt(seven_day_STD / availableDays);

                //Populating result with SMA & STD
                for (int back = row; back > row - availableDays; back--) {
                    result[back][4] = seven_day_SMA;
                    result[back][5] = seven_day_STD;
                }
            }

            //Represents downward pressure on the index
            double high_close_difference = data[row][4] - data[row][6];
            result[row][6] = high_close_difference;

            //Represents the upward pressure on the index
            double low_open_difference = data[row][5] - data[row][3];
            result[row][7] = low_open_difference;

            //Cumulative percentage change in the Close column from the first observation in the training set
            double cumulative_return = (data[row][6] - data[0][6]) / data[0][6];
            result[row][8] = cumulative_return;

            //EMA: exponential moving average of Close -- smoother and more responsive version of SMA
            double EMA_14_day = 0.0;
            const double SMOOTHING_FACTOR = 2.0/(1.0+14.0);
            if (row < 14) {
                //Initialize first 13 EMAs as SMAs
                for (int i = 0; i < row; i++) {
                    EMA_14_day += data[row][6];
                }
                EMA_14_day /= row;
            } else {
                EMA_14_day = (data[row][6] * SMOOTHING_FACTOR + result[row-1][9] * (1-SMOOTHING_FACTOR));
            }
            result[row][9] = EMA_14_day;

            // Close Change -- Difference between current close and previous day's close (similar to daily return)
            double close_change = (row > 0) ? data[row][6] - data[row-1][6] : 0.0;
            result[row][10] = close_change;

            // MACD -- Moving Average Convergence Divergence
            // - 12-Day EMA and 26-day EMA of Close % Change
            const double SMOOTHING_FACTOR_12 = 2.0/(1.0+12.0);
            const double SMOOTHING_FACTOR_26 = 2.0/(1.0+26.0);
            if (row < 12) { //Handle days < 12 case
                //Initialize first 13 EMAs as SMAs
                for (int i = 0; i < row; i++) {
                    EMA_12_day += data[row][6];
                }
                EMA_12_day /= row;
                result[row][11] = EMA_12_day;
            } else if (row < 26) { //Handle 12 < days < 26 case
                //Calculate the 12-day EMA
                EMA_12_day = (data[row][6] * SMOOTHING_FACTOR_12 + result[row-1][11] * (1-SMOOTHING_FACTOR_12));
                // Initialize 26-day EMA as an SMA
                for (int i = 0; i < row; i++) {
                    EMA_26_day += data[row][6];
                }
                EMA_26_day /= row;
                result[row][11] = EMA_26_day - EMA_12_day;
            } else {
                //Calculate both EMAs properly
                EMA_26_day = (data[row][6] * SMOOTHING_FACTOR_26 + EMA_26_day * (1-SMOOTHING_FACTOR_26));
                EMA_12_day = (data[row][6] * SMOOTHING_FACTOR_12 + EMA_12_day * (1-SMOOTHING_FACTOR_12));
                result[row][11] = EMA_26_day - EMA_12_day;
            }

            // Stochastic Oscillator over 14 days -- compares the Close with the High and Low columns over a 14-day window
            // Measures the position of the index relative to its recent range
            double stochastic_oscillator = 0.0;
            if (row < 14) {
                stochastic_oscillator = data[row][6]; //No current oscillation -- set as just the current close price
            } else {
                //Find the lowest and highest traded price of the previous 14 trading sessions -- i.e. max(High), min(Low)
                double lowest = data[row][5];
                double highest = data[row][4];
                for (int i = row; i > row - 14; i--) {
                    //Check for lowest
                    if (data[i][5] < lowest) {
                        lowest = data[i][5];
                    }
                    //Check for highest
                    if (data[i][4] > highest) {
                        highest = data[i][4];
                    }
                }

                //Calculate the stochastic oscillator
                stochastic_oscillator = (data[row][6] - lowest) / (highest - lowest);
            }
            result[row][12] = stochastic_oscillator;

            //Average true range (ATR) over 14 days -- Volatility indicator
            double ATR = 0.0;
            if (row == 0) {
                ATR = 0.0;
            } else if (row < 14) {
                for (int i = row; i > 0 && i > row - 14; i--) {
                    double true_range = std::max({data[i][4] - data[i][5], data[i][4] - data[i-1][6], data[i][5] - data[i-1][6]});
                    ATR += true_range;
                }
                ATR /= row;
            } else {
                ATR = (result[row-1][13] + std::max({data[row][4] - data[row][5], data[row][4] - data[row-1][6], data[row][5] - data[row-1][6]})) / 14;
            }

            result[row][13] = ATR;

            //Average directional index (ADX)
            double ADX = 0.0;
            double plus_DM = 0.0;
            double neg_DM = 0.0;
            if (row > 14) {

                for (int i = row; i > row - 14; i--) {
                    plus_DM += data[i][4] - data[i-1][4];
                    neg_DM += data[i-1][5] - data[i][5];
                }
                plus_DM = plus_DM - (plus_DM / 14) + (data[row][4] - data[row-1][4]);
                plus_DM /= result[row][13]; // Divide by ATR

                neg_DM = neg_DM - (neg_DM / 14) + (data[row-1][5] - data[row][5]);
                neg_DM /= result[row][13]; // Divide by ATR

                ADX = plus_DM - neg_DM; //ADX = difference of the two indicators
            }
            result[row][14] = ADX;

            // DMI/DX -- directional movement index: measures the positive and negative movements of the index
            double DMI = (plus_DM - neg_DM) / (plus_DM + neg_DM); //NOTE: DM here represent the DI (directional index)
            result[row][15] = DMI;
        }

        return result;
    }

    // Implement z-score normalization
    Matrix standardizeData(const Matrix& data) {
        Matrix result(data.size(), std::vector<double>(data[0].size(), 0.0));

        //Standardize the features across the columns
        for (int col = 0; col < data[0].size(); col++) {
            double stdev = 0.0;
            double mean = 0.0;
            // Calculate the mean across the column
            for (int row = 0; row < data.size(); row++) {
                mean += data[row][col];
            }
            mean /= data.size();;

            // Calculate the standard deviation across the column
            for (int row = 0; row < data.size(); row++) {
                stdev += std::pow(data[row][col] - mean, 2);
            }
            stdev = std::pow(stdev/data.size(), 0.5);

            //Z-Score Standardize:
            for (int row = 0; row < data.size(); row++) {
                if (stdev == 0) {
                    result[row][col] = 0.0; //Edge case: stdev in denominator = 0
                } else {
                    result[row][col] = (data[row][col] - mean) / stdev;
                }
            }
        }

        return result;
    }

    //Implement min-max normalization
    Matrix normalizeData(const Matrix& data) {
        Matrix result(data.size(), std::vector<double>(data[0].size(), 0.0));

        //Normalize the features across the columns
        for (int col = 0; col < data[0].size(); col++) {
            //Default values
            double min = data[0][col];
            double max = data[0][col];

            //Find the min and max values in the current column
            for (int row = 0; row < data.size(); row++) {
                min = std::min(min, data[row][col]);
                max = std::max(max, data[row][col]);
            }

            //Apply min-max normalization 
            for (int row = 0; row < data.size(); row++) {
                if (max - min == 0) {
                    result[row][col] = 0.5; //Edge case: max = min
                } else {
                    result[row][col] = (data[row][col] - min) / (max - min);
                }
            }
        }

        return result;
    }

    Tensor3D generate_tensor(const Matrix& data, const int timesteps) {
        Tensor3D result(data.size()-timesteps+1, Matrix(timesteps, std::vector<double>(data[0].size(), 0.0)));

        for (int example = 0; example < data.size()-timesteps+1; example++) {
            for (int t = 0; t < timesteps; t++) {
                for (int feature = 0; feature < data[0].size(); feature++) {
                    result[example][t][feature] = data[example + t][feature];
                }
            }
        }

        return result;
    }

    std::tuple<Tensor3D, Matrix> preprocessDataFromFile(const std::string& filename) {
        Matrix originalData = parseData(filename);
        const int TIMESTEPS = 30;

        Matrix x_matrix = normalizeData(standardizeData(engineerData(originalData)));

        Tensor3D x_train = generate_tensor(x_matrix, TIMESTEPS);

        originalData = normalizeData(standardizeData(originalData));
        Matrix y_train(originalData.size(), std::vector<double>(1, 0.0));
        for (int i = 0; i < originalData.size(); i++) {
            y_train[i][0] = originalData[i][6]; //Close column
        }

        return std::make_tuple(x_train, y_train);
    }

    Matrix preprocessData(const Matrix& data) {
        return normalizeData(standardizeData(engineerData(data)));
    }
}