#include <gunrock/applications/hits.hxx>
#include "hits_problem.hxx"
#include <gunrock/cuda/cuda.hxx>

#include <memory>
namespace gunrock{
namespace hits{

template<typename graph_t>
result_c<graph_t>& run(graph_t &G, int iter_times){
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  auto multi_context =
  std::shared_ptr<cuda::multi_context_t>(new cuda::multi_context_t(0));

  using problem_type = problem_t<graph_t>;
  using enactor_type = enactor_t<problem_type>;

  // qqq copy constructor
  result_c<graph_t>& result = new result_c<graph_t>;

  problem_type problem(G, multi_context);

  enactor_type enactor(problem, multi_context);
  enactor.enact();

  dump_result(result.get_auth(),
              result.get_hub(),
              problem.get_auth(),
              problem.get_hub());

  result.rank_auth();
  result.rank_hub();

  return result;
}

template<typename ForwardIterator>
void dump_result(ForwardIterator auth_dest,
                 ForwardIterator hub_dest,
                 ForwardIterator auth_src,
                 ForwardIterator hub_src){
  thrust::swap(auth_dest, auth_src);
  thrust::swap(hub_dest, hub_src);
}

}// namespace hits
}// namespace gunrock
