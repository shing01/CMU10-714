#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for(size_t i = 0; i < m; i += batch) {
        size_t current_batch_size = std::min(batch, m - i);
        // 计算 Z = X_batch * theta
        std::vector<float> Z(current_batch_size * k, 0.0f);
        for(size_t j = 0; j < current_batch_size; j++) { // 遍历 batch 每一行
            for(size_t col = 0; col < k; col++) { // 遍历 theta 每一列，即 k 个类别
                for(size_t dim = 0; dim < n; dim++) { // 遍历 theta 每一行，即 n 个输入维度
                    Z[j * k + col] += X[(i + j) * n + dim] * theta[dim * k + col];
                }
            }
        }
        // softmax
        std::vector<float> P(current_batch_size * k, 0.0f);
        for(size_t j = 0; j < current_batch_size; j++) {
            float sum_exp = 0.0f;
            for(size_t col = 0; col < k; col++) {
                P[j * k + col] = std::exp(Z[j * k + col]);
                sum_exp += P[j * k + col];
            }
            for(size_t col = 0; col < k; col++) {
                P[j * k + col] /= sum_exp;
            }
        }
        // SGD P-Y_One_Hot
        std::vector<float> Z_grad(current_batch_size * k, 0.0f);
        Z_grad = P;
        for(size_t j = 0; j < current_batch_size; j++) {
            unsigned char label = y[i + j];
            Z_grad[j * k + label] -= 1.0f;
        }
        float step = lr / (float)current_batch_size;
        for(size_t j = 0; j < current_batch_size; j++) {
            for(size_t col = 0; col < k; col++) {
                for(size_t dim = 0; dim < n; dim++) {
                    theta[dim * k + col] -= step * X[(i + j) * n + dim] * Z_grad[j * k + col];
                }
            }
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
