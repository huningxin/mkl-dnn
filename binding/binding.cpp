#include <emscripten/bind.h>

#include "mkldnn.h"

using namespace emscripten;

namespace utils {
    intptr_t mkldnn_engine_create_helper(int kind, size_t index) {
        mkldnn_engine_t aengine;
        mkldnn_engine_create(&aengine, (mkldnn_engine_kind_t)kind, index);
        return (intptr_t)aengine;
    }

    int mkldnn_engine_destroy_helper(intptr_t engine) {
        return (int)mkldnn_engine_destroy((mkldnn_engine_t)engine);
    }

    int mkldnn_memory_desc_init_helper(mkldnn_memory_desc_t* memory_desc, emscripten::val val, int data_type, int format) {

    }
}

EMSCRIPTEN_BINDINGS(mkldnn)
{
    constant("mkldnn_any_engine", (int)mkldnn_any_engine);
    constant("mkldnn_cpu", (int)mkldnn_cpu);

    function("mkldnn_engine_create", &utils::mkldnn_engine_create_helper, allow_raw_pointers());
    function("mkldnn_engine_destroy", &utils::mkldnn_engine_destroy_helper, allow_raw_pointers());

    class_<mkldnn_memory_desc_t>("mkldnn_memory_desc_t")
        .constructor<>()
        ;

    function("mkldnn_memory_desc_init", &utils::mkldnn_memory_desc_init);
}