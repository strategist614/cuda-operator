#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

// ============================================================
// Softmax Operator
//
// Input shape:
//   [dim0, dim1, ..., dimN]
//
// Softmax along:
//   axis
//
// Tensor is logically converted into:
//
//   [outer, softmax_dim, inner]
//
// where:
//
//   outer       = product(shape[0 : axis])
//   softmax_dim = shape[axis]
//   inner       = product(shape[axis + 1 : ])
//
// ============================================================

class Softmax {
public:
    explicit Softmax(int axis)
        : axis_(axis) {}

    void Forward(
        const float* input,
        float* output,
        const std::vector<int>& shape) const {

        // ----------------------------------------------------
        // 1. Validate input
        // ----------------------------------------------------
        if (input == nullptr) {
            throw std::invalid_argument("input is nullptr");
        }

        if (output == nullptr) {
            throw std::invalid_argument("output is nullptr");
        }

        if (shape.empty()) {
            throw std::invalid_argument("shape cannot be empty");
        }

        for (int dim : shape) {
            if (dim <= 0) {
                throw std::invalid_argument(
                    "all dimensions must be > 0");
            }
        }

        const int ndim = static_cast<int>(shape.size());

        // ----------------------------------------------------
        // 2. Normalize axis
        //
        // For example:
        // ndim = 4
        // axis = -1 -> 3
        // axis = -2 -> 2
        // ----------------------------------------------------
        int axis = axis_;

        if (axis < 0) {
            axis += ndim;
        }

        if (axis < 0 || axis >= ndim) {
            throw std::out_of_range("axis is out of range");
        }

        // ----------------------------------------------------
        // 3. Calculate outer / softmax_dim / inner
        // ----------------------------------------------------
        size_t outer = 1;

        for (int i = 0; i < axis; ++i) {
            outer *= static_cast<size_t>(shape[i]);
        }

        const size_t softmax_dim =
            static_cast<size_t>(shape[axis]);

        size_t inner = 1;

        for (int i = axis + 1; i < ndim; ++i) {
            inner *= static_cast<size_t>(shape[i]);
        }

        // ----------------------------------------------------
        // Layout:
        //
        // input[o, s, in]
        //
        // flattened index:
        //
        // index =
        //     o * softmax_dim * inner
        //   + s * inner
        //   + in
        //
        // ----------------------------------------------------

        for (size_t o = 0; o < outer; ++o) {

            for (size_t in = 0; in < inner; ++in) {

                // ====================================================
                // Step 1:
                // Reduce Max
                //
                // max_val = max(input[o, :, in])
                // ====================================================

                float max_val =
                    -std::numeric_limits<float>::infinity();

                for (size_t s = 0; s < softmax_dim; ++s) {

                    const size_t index =
                        o * softmax_dim * inner
                        + s * inner
                        + in;

                    max_val =
                        std::max(max_val, input[index]);
                }

                // ====================================================
                // Step 2:
                // exp(x - max) + Reduce Sum
                //
                // output temporarily stores:
                //
                // exp(input - max_val)
                // ====================================================

                float sum = 0.0f;

                for (size_t s = 0; s < softmax_dim; ++s) {

                    const size_t index =
                        o * softmax_dim * inner
                        + s * inner
                        + in;

                    const float exp_value =
                        std::exp(input[index] - max_val);

                    output[index] = exp_value;

                    sum += exp_value;
                }

                // ====================================================
                // Step 3:
                // Normalize
                //
                // y_i = exp(x_i - max) / sum
                // ====================================================

                const float inv_sum = 1.0f / sum;

                for (size_t s = 0; s < softmax_dim; ++s) {

                    const size_t index =
                        o * softmax_dim * inner
                        + s * inner
                        + in;

                    output[index] *= inv_sum;
                }
            }
        }
    }

private:
    int axis_;
};


// ============================================================
// Utility:
// Calculate number of tensor elements
// ============================================================

size_t NumElements(const std::vector<int>& shape) {

    size_t result = 1;

    for (int dim : shape) {
        result *= static_cast<size_t>(dim);
    }

    return result;
}


// ============================================================
// Utility:
// Print tensor recursively
// ============================================================

void PrintTensor(
    const std::vector<float>& data,
    const std::vector<int>& shape) {

    if (shape.empty()) {
        return;
    }

    const size_t total = NumElements(shape);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "[";

    for (size_t i = 0; i < total; ++i) {

        std::cout << data[i];

        if (i + 1 != total) {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";
}


// ============================================================
// Verify:
//
// Sum along softmax axis should be approximately 1
// ============================================================

void VerifySoftmax(
    const std::vector<float>& output,
    const std::vector<int>& shape,
    int axis) {

    const int ndim =
        static_cast<int>(shape.size());

    if (axis < 0) {
        axis += ndim;
    }

    size_t outer = 1;

    for (int i = 0; i < axis; ++i) {
        outer *= static_cast<size_t>(shape[i]);
    }

    const size_t softmax_dim =
        static_cast<size_t>(shape[axis]);

    size_t inner = 1;

    for (int i = axis + 1; i < ndim; ++i) {
        inner *= static_cast<size_t>(shape[i]);
    }

    std::cout << "\nVerify sum along axis:\n";

    for (size_t o = 0; o < outer; ++o) {

        for (size_t in = 0; in < inner; ++in) {

            float sum = 0.0f;

            for (size_t s = 0;
                 s < softmax_dim;
                 ++s) {

                const size_t index =
                    o * softmax_dim * inner
                    + s * inner
                    + in;

                sum += output[index];
            }

            std::cout
                << "outer=" << o
                << ", inner=" << in
                << ", sum=" << sum
                << '\n';
        }
    }
}


// ============================================================
// Main
// ============================================================

int main() {

    // --------------------------------------------------------
    // Example:
    //
    // shape = [2, 3, 4]
    //
    // axis = 1
    //
    // Means:
    //
    // For every [outer, inner] pair,
    // perform Softmax over 3 elements.
    //
    // outer       = 2
    // softmax_dim = 3
    // inner       = 4
    // --------------------------------------------------------

    const std::vector<int> shape = {
        2, 3, 4
    };

    const int axis = 1;

    const size_t num_elements =
        NumElements(shape);

    // --------------------------------------------------------
    // Input tensor
    //
    // shape:
    // [2, 3, 4]
    // --------------------------------------------------------

    std::vector<float> input = {

        // batch 0
        1.0f,  2.0f,  3.0f,  4.0f,
        2.0f,  4.0f,  6.0f,  8.0f,
        3.0f,  6.0f,  9.0f, 12.0f,

        // batch 1
        1.0f,  1.0f,  1.0f,  1.0f,
        2.0f,  2.0f,  2.0f,  2.0f,
        4.0f,  4.0f,  4.0f,  4.0f
    };

    if (input.size() != num_elements) {
        std::cerr
            << "Input size does not match shape.\n";

        return 1;
    }

    std::vector<float> output(
        num_elements,
        0.0f
    );

    // --------------------------------------------------------
    // Create Softmax operator
    // --------------------------------------------------------

    Softmax softmax(axis);

    // --------------------------------------------------------
    // Run
    // --------------------------------------------------------

    softmax.Forward(
        input.data(),
        output.data(),
        shape
    );

    // --------------------------------------------------------
    // Print
    // --------------------------------------------------------

    std::cout << "Input:\n";
    PrintTensor(input, shape);

    std::cout << "\nSoftmax output:\n";
    PrintTensor(output, shape);

    // --------------------------------------------------------
    // Verify every softmax group sums to 1
    // --------------------------------------------------------

    VerifySoftmax(
        output,
        shape,
        axis
    );

    return 0;
}