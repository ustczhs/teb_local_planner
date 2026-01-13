#ifndef FRENTE_REFERENCE_H
#define FRENTE_REFERENCE_H

#include <Eigen/Core>
#include <vector>

namespace teb_local_planner {

/**
 * @class FrenetReference
 * @brief Represents the reference line in Frenet coordinates (s, l)
 */
class FrenetReference {
public:
  /**
   * @brief Constructor
   * @param path_points Matrix where each row is [x, y, yaw, corridor_dis, ...]
   */
  FrenetReference(const Eigen::MatrixXd &path_points);

  /**
   * @brief Convert Cartesian point to Frenet coordinates
   * @param cart_point [x, y]
   * @return [s, l] Frenet coordinates
   */
  Eigen::Vector2d cartesianToFrenet(const Eigen::Vector2d &cart_point) const;

  /**
   * @brief Convert Frenet point to Cartesian coordinates
   * @param frenet_point [s, l]
   * @return [x, y] Cartesian coordinates
   */
  Eigen::Vector2d frenetToCartesian(const Eigen::Vector2d &frenet_point) const;

  /**
   * @brief Get corridor distance at a given s
   * @param s longitudinal coordinate
   * @return corridor distance
   */
  double getCorridorDistance(double s) const;

  /**
   * @brief Get reference point at index
   * @param index point index
   * @return [x, y] coordinates
   */
  Eigen::Vector2d getPoint(size_t index) const {
    if (index < reference_points_.size()) {
      return reference_points_[index];
    }
    return Eigen::Vector2d::Zero();
  }

  /**
   * @brief Get corridor distance at index
   * @param index point index
   * @return corridor distance
   */
  double getCorridorDistanceAtIndex(size_t index) const {
    if (index < corridor_distances_.size()) {
      return corridor_distances_[index];
    }
    return 1.5; // default
  }

  /**
   * @brief Get number of reference points
   * @return size
   */
  size_t size() const { return reference_points_.size(); }

private:
  std::vector<Eigen::Vector2d> reference_points_; // [x, y] for each point
  std::vector<double> cumulative_s_;              // cumulative s
  std::vector<double> corridor_distances_;        // corridor_dis for each point
};

} // namespace teb_local_planner

#endif // FRENTE_REFERENCE_H
