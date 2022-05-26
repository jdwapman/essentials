#include <tuple>

#define TILE_MATRIX 0
#define TILE_SPATIAL 1
#define TILE_TEMPORAL 2
#define TILE_TEMPORAL_BLOCK 3

// Forward declare
template <typename rowdim_t, typename coldim_t>
__host__ __device__ __forceinline__ constexpr auto make_layout(rowdim_t rowdim,
  coldim_t coldim);

template <typename rowdim_t, typename coldim_t, typename parentlayout_t>
__host__ __device__ __forceinline__ constexpr auto
make_layout(rowdim_t rowdim, coldim_t coldim, parentlayout_t parentlayout);

template <typename row_t, typename col_t>
struct Point {
  row_t row;
  col_t col;
  __host__ __device__ __forceinline__ Point(row_t _row, col_t _col)
    : row(_row), col(_col) {}
};

// TODO need to rewrite this to be runtime-modifiable (no tuples)
template <typename row_t, typename col_t, int hierarchy>
struct TileIdx {
  row_t row[hierarchy + 1];
  col_t col[hierarchy + 1];

  __host__ __device__ __forceinline__ TileIdx(row_t _row, col_t _col) {
    row[0] = _row;
    col[0] = _col;
  }

  __host__ __device__ __forceinline__ TileIdx(row_t _row,
    col_t _col,
    row_t* _parent_rows,
    col_t* _parent_cols) {
    row[hierarchy] = _row;
    col[hierarchy] = _col;
    for (int i = hierarchy - 1; i >= 0; i--) {
      row[i] = _parent_rows[i];
      col[i] = _parent_cols[i];
    }
  }

  __host__ __device__ __forceinline__ constexpr auto getHierarchy() const {
    return hierarchy;
  }
};

template <typename row_t, typename col_t>
__host__ __device__ __forceinline__ constexpr auto make_tile_index(row_t row,
  col_t col) {
  return TileIdx<row_t, col_t, 0>(row, col);
}

template <typename row_t, typename col_t, typename parenttile_t>
__host__ __device__ __forceinline__ constexpr auto
make_tile_index(row_t row, col_t col, parenttile_t parenttile) {
  return TileIdx<row_t, col_t, parenttile.getHierarchy() + 1>(
    row, col, parenttile.row, parenttile.col);
}

// NOTE: Need to store layout data as a tuple.
// Format: < <row0, col0>, <row1, col1>, ... >

template <typename tiledim_t>
class Layout {
public:
  __host__ __device__
    __forceinline__ constexpr Layout(const tiledim_t& _tiledims)
    : tiledims(_tiledims) {}

  template <typename rowdim_t, typename coldim_t>
  __host__ __device__ __forceinline__ constexpr auto tile(
    const rowdim_t rowdim,
    const coldim_t coldim) {
    return make_layout(rowdim, coldim, *this);
  }

  __host__ __device__ __forceinline__ constexpr bool has_parent() const {
    return std::tuple_size<tiledim_t>::value > 1;
  }

  __host__ __device__ __forceinline__ constexpr auto get_hierarchy_level()
    const {
    return std::tuple_size<tiledim_t>::value - 1;
  }

  // ===== TILE INFO FUNCTIONS ===== //

  // Get the dimensions of a tile
  __host__ __device__ __forceinline__ constexpr auto rows_in_tile(
    const int hierarchy) const {
    auto tiledim = TupleReturnValue(hierarchy, tiledims);
    return std::get<0>(tiledim);
  }

  __host__ __device__ __forceinline__ constexpr auto cols_in_tile(
    const int hierarchy) const {
    auto tiledim = TupleReturnValue(hierarchy, tiledims);
    return std::get<1>(tiledim);
  }

  template <typename tile_index_t>
  __host__ __device__ __forceinline__ constexpr auto rows_in_tile(
    const tile_index_t tile_index) const {
    return rows_in_tile(tile_index.getHierarchy());
  }

  template <typename tile_index_t>
  __host__ __device__ __forceinline__ constexpr auto cols_in_tile(
    const tile_index_t tile_index) const {
    return cols_in_tile(tile_index.getHierarchy());
  }

  // Get the number of child tiles
  // TODO need to handle the remainders
  __host__ __device__ __forceinline__ constexpr auto num_child_row_tiles(
    const int& hierarchy) const {
    if (hierarchy == get_hierarchy_level()) {
      return 1;
    }
    else {
      auto num_even_tiles =
        rows_in_tile(hierarchy) / rows_in_tile(hierarchy + 1);

      if (rows_in_tile(hierarchy) % rows_in_tile(hierarchy + 1) == 0) {
        return num_even_tiles;
      }
      else {
        return num_even_tiles + 1;
      }
    }
  }

  __host__ __device__ __forceinline__ constexpr auto num_child_col_tiles(
    const int hierarchy) const {
    if (hierarchy == get_hierarchy_level()) {
      return 1;
    }
    else {
      auto num_even_tiles =
        cols_in_tile(hierarchy) / cols_in_tile(hierarchy + 1);

      if (cols_in_tile(hierarchy) % cols_in_tile(hierarchy + 1) == 0) {
        return num_even_tiles;
      }
      else {
        return num_even_tiles + 1;
      }
    }
  }

  template <typename tile_index_t>
  __host__ __device__ __forceinline__ auto num_child_row_tiles(
    const tile_index_t& tile_index) const {
    return num_child_row_tiles(tile_index.getHierarchy());
  }

  template <typename tile_index_t>
  __host__ __device__ __forceinline__ auto num_child_col_tiles(
    const tile_index_t& tile_index) const {
    return num_child_col_tiles(tile_index.getHierarchy());
  }

  // Get the number of tiles at the level of the given tile
  __host__ __device__ __forceinline__ constexpr auto num_row_tiles_at_level(
    const int& hierarchy) const {
    if (hierarchy == 0) {
      return 1;
    }

    return num_child_row_tiles(hierarchy - 1);
  }

  __host__ __device__ __forceinline__ constexpr auto num_col_tiles_at_level(
    const int& hierarchy) const {
    if (hierarchy == 0) {
      return 1;
    }

    return num_child_col_tiles(hierarchy - 1);
  }

  template <typename tile_index_t>
  __host__ __device__ __forceinline__ constexpr auto num_row_tiles_at_level(
    const tile_index_t& tile_index) const {
    return num_row_tiles_at_level(tile_index.getHierarchy());
  }

  template <typename tile_index_t>
  __host__ __device__ __forceinline__ constexpr auto num_col_tiles_at_level(
    const tile_index_t& tile_index) const {
    return num_col_tiles_at_level(tile_index.getHierarchy());
  }

  // Not constexpr since the point changes at runtime
  template <typename point_t, typename tile_index_t, typename hierarchy_t>
  __host__ __device__ __forceinline__ auto remap_point(
    point_t point,
    tile_index_t tile_index,
    hierarchy_t goal_hierarchy) {
    if (tile_index.getHierarchy() < goal_hierarchy) {
      auto new_point = point;

#pragma unroll
      for (auto h_idx = tile_index.getHierarchy(); h_idx < goal_hierarchy;
        h_idx++) {
        new_point.row %= rows_in_tile(h_idx + 1);
        new_point.col %= cols_in_tile(h_idx + 1);
      }

      return new_point;

    }
    else if (tile_index.getHierarchy() > goal_hierarchy) {
      auto new_point = point;
      // Going from a small to a big tile
#pragma unroll
      for (auto h_idx = tile_index.getHierarchy(); h_idx > goal_hierarchy;
        h_idx--) {
        new_point.row += tile_index.row[h_idx] * rows_in_tile(h_idx);
        new_point.col += tile_index.col[h_idx] * cols_in_tile(h_idx);
      }

      return new_point;
    }
    else {
      return point;
    }
  }

  tiledim_t tiledims;
};

template <typename rowdim_t, typename coldim_t>
__host__ __device__ __forceinline__ constexpr auto make_layout(
  rowdim_t rowdim,
  coldim_t coldim) {
  std::tuple<rowdim_t, coldim_t> tiledim{ rowdim, coldim };
  std::tuple<decltype(tiledim)> tiledim_wrapper{ tiledim };
  return Layout<decltype(tiledim_wrapper)>(tiledim_wrapper);
}

template <typename rowdim_t, typename coldim_t, typename parentlayout_t>
__host__ __device__ __forceinline__ constexpr auto
make_layout(rowdim_t rowdim, coldim_t coldim, parentlayout_t parentlayout) {
  std::tuple<rowdim_t, coldim_t> tiledim{ rowdim, coldim };
  std::tuple<decltype(tiledim)> tiledim_wrapper{ tiledim };

  // concatenate parentlayout and tiledim tuples
  auto tiledim_wrapper_nested =
    std::tuple_cat(parentlayout.tiledims, tiledim_wrapper);

  return Layout<decltype(tiledim_wrapper_nested)>(tiledim_wrapper_nested);
}