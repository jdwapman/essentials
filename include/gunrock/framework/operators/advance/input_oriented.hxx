/**
 * @file input_oriented.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-20
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/util/math.hxx>
#include <gunrock/cuda/context.hxx>

#include <gunrock/framework/operators/configs.hxx>
#include <gunrock/framework/operators/for/for.hxx>

#include <thrust/transform_scan.h>
#include <thrust/iterator/discard_iterator.h>

namespace gunrock {
namespace operators {
namespace advance {
namespace input_oriented {

template <typename graph_t, typename operator_t, typename type_t>
__global__ void input_oriented_kernel(const graph_t G,
                                      operator_t op,
                                      type_t* input,
                                      const std::size_t input_size,
                                      type_t* output,
                                      const std::size_t output_size) {
  for (int idx = threadIdx.x + (blockIdx.x * blockDim.x);  // Index into Input
       idx < input_size;                                   // Bound checking
       idx += blockDim.x * gridDim.x                       // Stride
  ) {
    auto v = input[idx];
    if (!gunrock::util::limits::is_valid(v))
      continue;

    auto starting_edge = G.get_starting_edge(v);
    auto total_edges = G.get_number_of_neighbors(v);

    for (auto e = starting_edge; e < starting_edge + total_edges; ++e) {
      auto n = G.get_destination_vertex(e);
      auto w = G.get_edge_weight(e);
      bool cond = op(v, n, e, w);
      output[e] = (cond && n != v)
                      ? n
                      : gunrock::numeric_limits<decltype(v)>::invalid();
    }
  }
}

template <advance_type_t type,
          advance_direction_t direction,
          typename graph_t,
          typename operator_t,
          typename frontier_t,
          typename work_tiles_t>
void execute(graph_t& G,
             operator_t op,
             frontier_t* input,
             frontier_t* output,
             work_tiles_t& segments,
             cuda::standard_context_t& context) {
  using vertex_t = typename graph_t::vertex_type;

  auto size_of_output = compute_output_length(G, input, segments, context);

  // If output frontier is empty, resize and return.
  if (size_of_output <= 0) {
    output->set_number_of_elements(0);
    return;
  }

  // <todo> Resize the output (inactive) buffer to the new size.
  // Can be hidden within the frontier struct.
  // if (output->get_capacity() < size_of_output)
  output->resize(size_of_output);
  output->set_number_of_elements(size_of_output);
  // </todo>

  // Get output data of the active buffer.
  auto output_data = output->data();

  auto pre_condition = [=] __device__(vertex_t const& v) {
    return gunrock::util::limits::is_valid(v);
  };

  auto neighbors_expand = [=] __device__(vertex_t const& v) {
    auto starting_edge = G.get_starting_edge(v);
    auto total_edges = G.get_number_of_neighbors(v);

    for (auto e = starting_edge; e < starting_edge + total_edges; ++e) {
      auto n = G.get_destination_vertex(e);
      auto w = G.get_edge_weight(e);
      bool cond = op(v, n, e, w);
      output_data[e] =
          (cond && n != v) ? n : gunrock::numeric_limits<vertex_t>::invalid();
    }

    return v;  // output is discarded.
  };

  output->fill(gunrock::numeric_limits<vertex_t>::invalid());

  thrust::transform_if(
      thrust::cuda::par.on(context.stream()),  // execution policy
      input->begin(),                          // input iterator: first
      input->end(),                            // input iterator: last
      thrust::make_discard_iterator(),         // output iterator: ignore
      neighbors_expand,                        // unary operation
      pre_condition                            // predicate operation
  );
}
}  // namespace input_oriented
}  // namespace advance
}  // namespace operators
}  // namespace gunrock