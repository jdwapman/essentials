#include <nlohmann/json.hpp>

// for convenience
using json = nlohmann::json;

template <typename args_t>
void log_cmd_args(json &_json, args_t args)
{
  auto args_vec = args.arguments();

  // Iterate over the arguments
  _json["argc"] = args_vec.size();

  for (auto arg : args_vec)
  {
    _json["argv"][arg.key()] = arg.value();
  }
}