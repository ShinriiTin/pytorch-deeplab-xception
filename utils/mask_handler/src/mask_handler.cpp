#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <queue>
#include <utility>
#include <vector>

namespace py = pybind11;

using int64 = long long;

std::pair<py::array_t<int64>, std::string>
get_rail_from_mask(const py::array_t<int64>& mask) {
    py::buffer_info buf = mask.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("numpy.ndarry dims must be 2!");
    }

    auto new_mask = py::array_t<int64>(buf.size);
    new_mask.resize({buf.shape[0], buf.shape[1]});

    py::buffer_info new_buf = new_mask.request();

    int64* ptr1 = (int64*)buf.ptr;
    int64* ptr2 = (int64*)new_buf.ptr;
    std::fill_n(ptr2, buf.size, 0);

    std::pair<int, int> start = {-1, -1};
    std::pair<int, int> camera = {buf.shape[0], buf.shape[1] / 2};
    int min_dist = -1;
    for (int i = 0; i < buf.shape[0]; ++i) {
        for (int j = 0; j < buf.shape[1]; ++j) {
            if (ptr1[i * buf.shape[1] + j] == 0) {
                continue;
            }
            int dist =
                (i - camera.first) * (i - camera.first) +
                (j - camera.second) * (j - camera.second);
            if (min_dist < 0 || min_dist > dist) {
                start = {i, j};
                min_dist = dist;
            }
        }
    }
    if (min_dist == -1) {
        return {new_mask, "unknown"};
    }

    std::queue<std::pair<int, int>> Q;
    int start_pos = start.first * buf.shape[1] + start.second;
    ptr2[start_pos] = ptr1[start_pos];
    Q.push(start);
    while (!Q.empty()) {
        auto cur = Q.front();
        Q.pop();
        int row = cur.first;
        int col = cur.second;
        if (row > 0) {
            int npos = (row - 1) * buf.shape[1] + col;
            if (ptr2[npos] != ptr1[npos]) {
                ptr2[npos] = ptr1[npos];
                Q.emplace(row - 1, col);
            }
        }
        if (col > 0) {
            int npos = row * buf.shape[1] + col - 1;
            if (ptr2[npos] != ptr1[npos]) {
                ptr2[npos] = ptr1[npos];
                Q.emplace(row, col - 1);
            }
        }
        if (col + 1 < buf.shape[1]) {
            int npos = row * buf.shape[1] + col + 1;
            if (ptr2[npos] != ptr1[npos]) {
                ptr2[npos] = ptr1[npos];
                Q.emplace(row, col + 1);
            }
        }
    }

    std::vector<std::pair<int, int>> curve;
    for (int i = buf.shape[0] - 1; ~i; --i) {
        int min_x = -1;
        for (int j = 0; j < buf.shape[1]; ++j) {
            if (ptr2[i * buf.shape[1] + j] > 0) {
                min_x = j;
                break;
            }
        }
        if (min_x == -1) {
            continue;
        }
        int max_x = -1;
        for (int j = buf.shape[1] - 1; ~j; --j) {
            if (ptr2[i * buf.shape[1] + j] > 0) {
                max_x = j;
                break;
            }
        }
        curve.emplace_back(min_x + max_x, 2 * i);
    }

    if (curve.size() < 20) {
        return {new_mask, "unknown"};
    }
    std::pair<int, int> start_direction =
        {curve[8].first - curve[4].first + curve[14].first - curve[10].first,
         curve[8].second - curve[4].second + curve[14].second - curve[10].second};
    constexpr double lmt = std::cos(std::acos(-1.) / 12);
    for (int i = 20; i + 20 < curve.size(); i += 20) {
        std::pair<int, int> direction =
            {curve[i + 8].first - curve[i + 4].first + curve[i + 14].first - curve[i + 10].first,
             curve[i + 8].second - curve[i + 4].second + curve[i + 14].second - curve[i + 10].second};
        double cos_alpha =
            (start_direction.first * direction.first + start_direction.second * direction.second) /
            std::sqrt(
              (start_direction.first * start_direction.first + start_direction.second * start_direction.second)
              * (direction.first * direction.first + direction.second * direction.second));
        if (cos_alpha >= lmt) {
            continue;
        }
        int cross = direction.first * start_direction.second - direction.second * start_direction.first;
        if (cross > 0) {
            return {new_mask, "left"};
        } else {
            return {new_mask, "right"};
        }
    }
    return {new_mask, "straight"};
}

PYBIND11_MODULE(mask_handler, m) {
    m.doc() = "mask handler using c++";
    m.def("get_rail_from_mask", &get_rail_from_mask);
}