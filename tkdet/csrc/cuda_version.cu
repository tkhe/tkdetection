#include <cuda_runtime_api.h>

namespace tkdet {
    int get_cudart_version() {
        return CUDART_VERSION;
    }
}
