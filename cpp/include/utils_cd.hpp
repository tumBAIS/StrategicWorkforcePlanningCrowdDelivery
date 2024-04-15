#ifndef CD_CPP_UTILS_HPP
#define CD_CPP_UTILS_HPP

#include "npy.hpp"

struct NPYArray1D {
    static const unsigned int dim = 1;
    std::vector<unsigned long> shape;
    std::vector<double> data;
    NPYArray1D() : shape(), data() {};
    [[nodiscard]] auto get(size_t i) const -> double {
        return this->data[i];
    }
    auto at(size_t i) -> double& {
        return this->data[i];
    }
};

struct NPYArray2D {
    static const unsigned int dim = 2;
    std::vector<unsigned long> shape;
    std::vector<double> data;
    NPYArray2D() : shape(), data() {};
    [[nodiscard]] auto get(size_t i,size_t j) const -> double {
        return this->data[i*this->shape[1]+j];
    }
    auto at(size_t i,size_t j) -> double& {
        return this->data[i*this->shape[1]+j];
    }
    auto copy_row(size_t i) -> NPYArray1D {
        auto arr = NPYArray1D();
        arr.shape.push_back(shape[1]);
        for (auto j = 0; j < shape[1]; ++j) {
            arr.data.push_back(data[i * shape[1] + j]);
        }
        return arr;
    }
};

struct NPYArray3D {
    static const unsigned int dim = 3;
    std::vector<unsigned long> shape;
    std::vector<double> data;
    NPYArray3D() : shape(), data() {};
    auto at(size_t i,size_t j, size_t k) -> double& {
        return this->data[(i*this->shape[2]*this->shape[1])+(j*this->shape[2])+k];
    }
};

struct NPYArray4D {
    static const unsigned int dim = 4;
    std::vector<unsigned long> shape;
    std::vector<double> data;
    NPYArray4D() : shape(), data() {};
    auto at(size_t i,size_t j, size_t k, size_t l) -> double& {
        return this->data[(i*this->shape[3]*this->shape[2]*this->shape[1])+(j*this->shape[3]*this->shape[2])+k*this->shape[3]+l];
    }
};

template<typename T>
auto load_numpy_array(const std::string &filename) -> T {
    auto array = T();
    bool fortran_order = false;
    npy::LoadArrayFromNumpy(filename, array.shape, fortran_order, array.data);
    return array;
}

template<typename T>
auto save_numpy_array(const std::string &filename, T& array) {
    npy::SaveArrayAsNumpy(filename, false, T::dim, array.shape.data(), array.data.data());
}

auto create_npy_array_4d(std::initializer_list<int> shape) -> NPYArray4D {
    auto array = NPYArray4D();
    array.shape = std::vector<unsigned long>(4);

    auto it = shape.begin();
    for (auto i = 0; i < 4; ++i) {
        assert(it != shape.end());
        array.shape[i] = *it;
        ++it;
    }
    assert(it == shape.end());

    array.data = std::vector<double>(array.shape[0] * (array.shape[3] * array.shape[2] * array.shape[1]));

    return array;
}


#endif //CD_CPP_UTILS_HPP
