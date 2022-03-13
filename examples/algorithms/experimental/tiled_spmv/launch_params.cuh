#include <gunrock/cuda/launch_box.hxx>

using namespace gunrock::cuda::launch_box;

/* ========== SM_80 ========== */

// Known runtime constants
#define SM80_l2CacheSize 0

// Persisting cache parameters
#define SM80_persistingL2CacheMaxSize 0
#define SM80_accessPolicyMaxWindowSize 0

// Grid parameters
#define SM80_maxThreadsPerBlock 0
#define SM80_sharedMemPerBlockOptin 0
#define SM80_multiProcessorCount 0

/* ========== SM_86 ========== */

// Known runtime constants
#define SM86_l2CacheSize 0

// Persisting cache parameters
#define SM86_persistingL2CacheMaxSize 0
#define SM86_accessPolicyMaxWindowSize 0

// Grid parameters
#define SM86_maxThreadsPerBlock 0
#define SM86_sharedMemPerBlockOptin 0
#define SM86_multiProcessorCount 0

template <unsigned int l2CacheSize_ = 0,
          unsigned int persistingL2CacheMaxSize_ = 0,
          unsigned int accessPolicyMaxWindowSize_ = 0,
          unsigned int maxThreadsPerBlock_ = 0,
          unsigned int sharedMemPerBlockOptin_ = 0,
          unsigned int multiProcessorCount_ = 0>
struct device_params_t {
  // Save the template unsigned ints so we can get them at compile time
  static constexpr unsigned int l2CacheSize = l2CacheSize_;
  static constexpr unsigned int persistingL2CacheMaxSize =
      persistingL2CacheMaxSize_;
  static constexpr unsigned int accessPolicyMaxWindowSize =
      accessPolicyMaxWindowSize_;
  static constexpr unsigned int maxThreadsPerBlock = maxThreadsPerBlock_;
  static constexpr unsigned int sharedMemPerBlockOptin =
      sharedMemPerBlockOptin_;
  static constexpr unsigned int multiProcessorCount = multiProcessorCount_;
};

typedef device_params_t<SM80_l2CacheSize,
                        SM80_persistingL2CacheMaxSize,
                        SM80_accessPolicyMaxWindowSize,
                        SM80_maxThreadsPerBlock,
                        SM80_sharedMemPerBlockOptin,
                        SM80_multiProcessorCount>
    device_params_80;

typedef device_params_t<SM86_l2CacheSize,
                        SM86_persistingL2CacheMaxSize,
                        SM86_accessPolicyMaxWindowSize,
                        SM86_maxThreadsPerBlock,
                        SM86_sharedMemPerBlockOptin,
                        SM86_multiProcessorCount>
    device_params_86;

template <sm_flag_t sm_flags_,
          typename block_dimensions_,
          typename grid_dimensions_,
          typename device_props_,
          std::size_t items_per_thread_ = 1,
          std::size_t shared_memory_bytes_ = 0>
struct tiled_launch_params_t : launch_params_t<sm_flags_,
                                               block_dimensions_,
                                               grid_dimensions_,
                                               items_per_thread_,
                                               shared_memory_bytes_> {
  typedef launch_params_t<sm_flags_,
                          block_dimensions_,
                          grid_dimensions_,
                          items_per_thread_,
                          shared_memory_bytes_>
      launch_params_t;

  typedef device_props_ device_props_t;
};

// Block configuration
// TODO fill this in with values determined by running the program with
// runtime configuration
typedef launch_box_t<tiled_launch_params_t<sm_86,             // 3070
                                           dim3_t<64, 1, 1>,  // Block dim
                                           dim3_t<64, 1, 1>,  // Grid dim
                                           device_params_86,  // Device props
                                           2,                 // Items
                                           1>,                // Shared mem
                     tiled_launch_params_t<sm_80,             // A100
                                           dim3_t<64, 1, 1>,
                                           dim3_t<64, 1, 1>,
                                           device_params_80,
                                           2,
                                           1>>
    launch_t;
