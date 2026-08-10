#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;
void softmax1D(
    const std::vector<float>& input,
    std::vector<float>& output) 
{
    const size_t n = input.size();
    output.resize(n);

    float max_val = -std::numeric_limits<float>::infinity();
    
    for(auto val : input) {
        max_val = std::max(max_val, val);
    }

    float sum = 0.0f;
    for(auto val : input) {
        sum += std::exp(val - max_val);
    }   

    for(size_t i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_val) / sum;
    }

}

int main()
{
    std::vector<float> input = {
        1.0f,
        2.0f,
        3.0f,
        4.0f
    };

    std::vector<float> output;

    softmax1D(input, output);

    std::cout << "Softmax output:\n";

    for (float value : output) {
        std::cout << value << " ";
    }

    std::cout << '\n';

    return 0;
}