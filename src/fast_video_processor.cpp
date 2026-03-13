#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<float> compute_physics_flow_cpp(py::object prev_obj,
                                            py::object curr_obj,
                                            int flow_size) {
  std::cout << "[CPP] Starting compute_physics_flow_cpp (MINIMAL MODE)"
            << std::endl;

  std::vector<ssize_t> shape = {flow_size, flow_size, 4};
  py::array_t<float> result(shape);

  std::cout << "[CPP] Array allocated. Returning." << std::endl;
  return result;
}

PYBIND11_MODULE(fast_video_processor, m) {
  m.doc() = "C++ Minimal Video Processing Module using PyBind11";
  m.def("compute_physics_flow_cpp", &compute_physics_flow_cpp,
        "Compute 4-channel physics flow (MINIMAL DUMMY)", py::arg("prev_bgr"),
        py::arg("curr_bgr"), py::arg("flow_size"));
}
