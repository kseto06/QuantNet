#ifndef DATAFRAMEWORK_H
#define DATAFRAMEWORK_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <tuple>

namespace DataFramework {
    // Type definitions
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::vector<std::vector<std::vector<double>>> Tensor3D;

    // Function declarations
    Matrix parseData(const std::string& filename);
    time_t UnixTimestamp(const double year, const double month, const double day);
    Matrix engineerData(const Matrix& data);
    Matrix standardizeData(const Matrix& data);
    Matrix normalizeData(const Matrix& data);
    std::tuple<Tensor3D, Matrix> preprocessDataFromFile(const std::string& filename);
    Matrix preprocessData(const Matrix& data);
}

#endif
