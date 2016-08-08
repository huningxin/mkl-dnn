#ifndef CPU_REFERENCE_INNER_PRODUCT_HPP
#define CPU_REFERENCE_INNER_PRODUCT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_engine.hpp"
#include "primitive.hpp"
#include "type_helpers.hpp"

namespace mkl_dnn {
namespace impl {
namespace cpu {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::prop_kind;
using namespace mkl_dnn::impl::primitive_kind;

template <impl::precision_t prec>
class reference_inner_product : public primitive {
private:
    const impl::inner_product_primitive_desc_t &_ippd;

    status_t execute_forward();
    status_t execute_backward_data();
    status_t execute_backward_weights();

protected:
    status_t execute_impl()
    {
        status_t status = success;
        _exec_state = busy;
        switch (_ippd.inner_product_desc.prop_kind) {
        case forward: status = execute_forward(); break;
        case backward_data: status = execute_backward_data(); break;
        case backward_weights: status = execute_backward_weights(); break;
        default:
            assert(0 && "invalid prop_kind"); // should never happen
        }
        _exec_state = done;
        return status;
    }

public:
    typedef typename precision2type<prec>::type data_t;

    reference_inner_product(const inner_product_primitive_desc_t &ippd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(
                  ippd, const_cast<impl::engine *>(ippd.base.engine), not_ready)
        , _ippd(_primitive_desc.inner_product)
    {
        for (int i = 0; i < 2; ++i)
            _input.push_back(inputs[i]);
        _output.push_back(outputs[0]);
    }
    ~reference_inner_product() {}

    /* static magic */
    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkl_dnn::impl::engine &aengine);
    static const primitive_impl implementation;
};
}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s