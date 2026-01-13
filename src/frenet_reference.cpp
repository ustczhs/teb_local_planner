#include "frenet_reference.h"
#include <cmath>

namespace teb_local_planner {

FrenetReference::FrenetReference(const Eigen::MatrixXd &path_points) {
  if (path_points.cols() < 4) {
    throw std::invalid_argument(
        "path_points must have at least 4 columns: x, y, yaw, corridor_dis");
  }

  reference_points_.reserve(path_points.rows());
  cumulative_s_.reserve(path_points.rows());
  corridor_distances_.reserve(path_points.rows());

  double s_accum = 0.0;
  cumulative_s_.push_back(0.0);

  for (int i = 0; i < path_points.rows(); ++i) {
    Eigen::Vector2d point(path_points(i, 0), path_points(i, 1));
    reference_points_.push_back(point);
    corridor_distances_.push_back(path_points(i, 3)); // corridor_dis

    if (i > 0) {
      double dist = (point - reference_points_[i - 1]).norm();
      s_accum += dist;
      cumulative_s_.push_back(s_accum);
    }
  }
}

Eigen::Vector2d
FrenetReference::cartesianToFrenet(const Eigen::Vector2d &cart_point) const {
  if (reference_points_.empty()) {
    return Eigen::Vector2d::Zero();
  }

  // Find the closest point on the reference line
  int closest_idx = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (size_t i = 0; i < reference_points_.size(); ++i) {
    double dist = (cart_point - reference_points_[i]).norm();
    if (dist < min_dist) {
      min_dist = dist;
      closest_idx = i;
    }
  }

  Eigen::Vector2d ref_point = reference_points_[closest_idx];
  double s = cumulative_s_[closest_idx];

  // Calculate lateral distance (l)
  Eigen::Vector2d direction;
  if (closest_idx < reference_points_.size() - 1) {
    direction = reference_points_[closest_idx + 1] - ref_point;
  } else if (closest_idx > 0) {
    direction = ref_point - reference_points_[closest_idx - 1];
  } else {
    // Single point, assume horizontal
    direction = Eigen::Vector2d(1.0, 0.0);
  }

  direction.normalize();
  Eigen::Vector2d normal(-direction.y(), direction.x()); // perpendicular vector

  double l = (cart_point - ref_point).dot(normal);

  return Eigen::Vector2d(s, l);
}

Eigen::Vector2d
FrenetReference::frenetToCartesian(const Eigen::Vector2d &frenet_point) const {
  double s = frenet_point.x();
  double l = frenet_point.y();

  // Find the segment containing s
  size_t idx = 0;
  for (size_t i = 1; i < cumulative_s_.size(); ++i) {
    if (cumulative_s_[i] >= s) {
      idx = i - 1;
      break;
    }
    idx = i;
  }

  Eigen::Vector2d ref_point = reference_points_[idx];
  Eigen::Vector2d direction;

  if (idx < reference_points_.size() - 1) {
    direction = reference_points_[idx + 1] - ref_point;
  } else if (idx > 0) {
    direction = ref_point - reference_points_[idx - 1];
  } else {
    direction = Eigen::Vector2d(1.0, 0.0);
  }

  direction.normalize();
  Eigen::Vector2d normal(-direction.y(), direction.x());

  Eigen::Vector2d cart_point = ref_point + l * normal;
  return cart_point;
}

double FrenetReference::getCorridorDistance(double s) const {
  if (corridor_distances_.empty()) {
    return 1.5; // default
  }

  // Interpolate corridor distance
  size_t idx = 0;
  for (size_t i = 1; i < cumulative_s_.size(); ++i) {
    if (cumulative_s_[i] >= s) {
      idx = i - 1;
      break;
    }
    idx = i;
  }

  if (idx >= corridor_distances_.size() - 1) {
    return corridor_distances_.back();
  }

  double s1 = cumulative_s_[idx];
  double s2 = cumulative_s_[idx + 1];
  double d1 = corridor_distances_[idx];
  double d2 = corridor_distances_[idx + 1];

  if (s2 == s1) {
    return d1;
  }

  double ratio = (s - s1) / (s2 - s1);
  return d1 + ratio * (d2 - d1);
}

} // namespace teb_local_planner
