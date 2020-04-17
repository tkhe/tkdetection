#include <torch/extension.h>
#include "ROIAlign/ROIAlign.h"
#include "deformable/deform_conv.h"

namespace tkdet {

    #ifdef WITH_CUDA
    extern int get_cudart_version();
    #endif

    std::string get_cuda_version() {
        #ifdef WITH_CUDA
        std::ostringstream oss;

        auto printCudaStyleVersion = [&](int v) {
            oss << (v / 1000) << "." << (v / 10 % 100);
            if (v % 10 != 0) {
                oss << "." << (v % 10);
            }
        };
        printCudaStyleVersion(get_cudart_version());
        return oss.str();
        #else
        return std::string("not available");
        #endif
    }

    std::string get_compiler_version() {
        std::ostringstream ss;
        #if defined(__GNUC__)
        #ifndef __clang__

        #if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
        #error "GCC >= 4.9 is required!"
        #endif

        { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
        #endif
        #endif

        #if defined(__clang_major__)
        {
            ss << "clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
        }
        #endif

        #if defined(_MSC_VER)
        { ss << "MSVC " << _MSC_FULL_VER; }
        #endif
        return ss.str();
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
        m.def("get_cuda_version", &get_cuda_version, "get_cuda_version");

        m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
        m.def(
            "deform_conv_backward_input",
            &deform_conv_backward_input,
            "deform_conv_backward_input"
        );
        m.def(
            "deform_conv_backward_filter",
            &deform_conv_backward_filter,
            "deform_conv_backward_filter"
        );
        m.def(
            "modulated_deform_conv_forward",
            &modulated_deform_conv_forward,
            "modulated_deform_conv_forward"
        );
        m.def(
            "modulated_deform_conv_backward",
            &modulated_deform_conv_backward,
            "modulated_deform_conv_backward"
        );

        m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
        m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
    }

}
