#ifndef EDGE_FRENTE_CORRIDOR_H
#define EDGE_FRENTE_CORRIDOR_H

#include "base_teb_edges.h"
#include "frenet_reference.h"
#include "penalties.h"
#include "vertex_pose.h"
#include <Eigen/Core>
#include <limits>

namespace teb_local_planner {

/**
 * @class EdgeFrenetCorridor
 * @brief Edge defining the cost function for keeping the robot within the
 * Frenet corridor (implemented in Cartesian coordinates)
 *
 * The edge depends on a single vertex \f$ \mathbf{s}_i \f$ and minimizes:
 * \f$ \min \rho(d) \cdot weight \f$, where \f$ d = |dist| - corridor_{width}/2
 * \f$ if \f$ |dist| > corridor_{width}/2 \f$, and \f$ \rho(d) \f$ is a penalty
 * function.
 */
class EdgeFrenetCorridor : public BaseTebUnaryEdge<1, double, VertexPose> {
public:
  /**
   * @brief Construct edge.
   */
  EdgeFrenetCorridor() { this->setMeasurement(0.); }

  /**
   * @brief Actual cost function (Cartesian coordinates)
   */
  void computeError() {
    const VertexPose *pose_vertex =
        static_cast<const VertexPose *>(_vertices[0]);
    const Eigen::Vector2d pose_cart(pose_vertex->pose().position().x(),
                                    pose_vertex->pose().position().y());

    // Find closest point on reference line
    Eigen::Vector2d closest_point;
    double min_dist = std::numeric_limits<double>::max();
    size_t closest_idx = 0;

    for (size_t i = 0; i < frenet_ref_->size(); ++i) {
      Eigen::Vector2d ref_point = frenet_ref_->getPoint(i);
      double dist = (pose_cart - ref_point).norm();
      if (dist < min_dist) {
        min_dist = dist;
        closest_point = ref_point;
        closest_idx = i;
      }
    }

    // Calculate signed distance to reference line
    Eigen::Vector2d direction;
    if (closest_idx < frenet_ref_->size() - 1) {
      direction = frenet_ref_->getPoint(closest_idx + 1) - closest_point;
    } else if (closest_idx > 0) {
      direction = closest_point - frenet_ref_->getPoint(closest_idx - 1);
    } else {
      direction = Eigen::Vector2d(1.0, 0.0); // default
    }

    direction.normalize();
    Eigen::Vector2d normal(-direction.y(), direction.x()); // perpendicular
    double signed_dist = (pose_cart - closest_point).dot(normal);

    // Get corridor width from reference
    double corridor_width =
        frenet_ref_->getCorridorDistanceAtIndex(closest_idx);
    double half_width = corridor_width / 2.0;

    // Soft penalty for exceeding corridor bounds
    double error = 0.0;
    if (std::abs(signed_dist) > half_width) {
      error = std::abs(signed_dist) - half_width;
    }

    _error[0] = penaltyBoundFromBelow(error, 0.0, 0.0); // soft constraint
  }

  /**
   * @brief Set Frenet reference
   * @param frenet_ref Shared pointer to FrenetReference
   */
  void setFrenetReference(boost::shared_ptr<FrenetReference> frenet_ref) {
    frenet_ref_ = frenet_ref;
  }

  /**
   * @brief Set corridor weight
   * @param weight Weight for the edge
   */
  void setCorridorWeight(double weight) { _information(0, 0) = weight; }

  /** @brief Return the error of the edge. */
  void linearizeOplus() {
    const VertexPose *pose_vertex =
        static_cast<const VertexPose *>(_vertices[0]);
    const Eigen::Vector2d pose_cart(pose_vertex->pose().position().x(),
                                    pose_vertex->pose().position().y());

    // Find closest point and direction
    Eigen::Vector2d closest_point;
    double min_dist = std::numeric_limits<double>::max();
    size_t closest_idx = 0;

    for (size_t i = 0; i < frenet_ref_->size(); ++i) {
      Eigen::Vector2d ref_point = frenet_ref_->getPoint(i);
      double dist = (pose_cart - ref_point).norm();
      if (dist < min_dist) {
        min_dist = dist;
        closest_point = ref_point;
        closest_idx = i;
      }
    }

    Eigen::Vector2d direction;
    if (closest_idx < frenet_ref_->size() - 1) {
      direction = frenet_ref_->getPoint(closest_idx + 1) - closest_point;
    } else if (closest_idx > 0) {
      direction = closest_point - frenet_ref_->getPoint(closest_idx - 1);
    } else {
      direction = Eigen::Vector2d(1.0, 0.0);
    }
    direction.normalize();

    Eigen::Vector2d normal(-direction.y(), direction.x());
    Eigen::Vector2d to_point = pose_cart - closest_point;
    double signed_dist = to_point.dot(normal);

    double corridor_width =
        frenet_ref_->getCorridorDistanceAtIndex(closest_idx);
    double half_width = corridor_width / 2.0;

    double error = 0.0;
    if (std::abs(signed_dist) > half_width) {
      error = std::abs(signed_dist) - half_width;
    }

    double penalty_deriv = penaltyBoundFromBelowDerivative(error, 0.0, 0.0);

    // Jacobian w.r.t. x, y: derivative of signed distance
    if (std::abs(signed_dist) > half_width) {
      double sign = (signed_dist > 0) ? 1.0 : -1.0;
      _jacobianOplusXi(0, 0) = penalty_deriv * sign * normal.x();
      _jacobianOplusXi(0, 1) = penalty_deriv * sign * normal.y();
    } else {
      _jacobianOplusXi(0, 0) = 0.0;
      _jacobianOplusXi(0, 1) = 0.0;
    }

    _jacobianOplusXi(0, 2) = 0.0; // dyaw (no direct effect)
  }

  /**
   * @brief 判断点是否在走廊边界内
   * @param point 要判断的点（世界坐标系）
   * @return true-点在走廊内，false-点在走廊外或无走廊参考
   */
  bool isPointInCorridor(const Eigen::Vector2d &point) const {
    // 如果没有Frenet参考轨迹，返回false
    if (!frenet_ref_ || frenet_ref_->size() == 0) {
      return false;
    }
    
    // 找到最近的点
    Eigen::Vector2d closest_point;
    double min_dist = std::numeric_limits<double>::max();
    size_t closest_idx = 0;
    
    for (size_t i = 0; i < frenet_ref_->size(); ++i) {
      Eigen::Vector2d ref_point = frenet_ref_->getPoint(i);
      double dist = (point - ref_point).norm();
      if (dist < min_dist) {
        min_dist = dist;
        closest_point = ref_point;
        closest_idx = i;
      }
    }
    
    // 计算方向向量
    Eigen::Vector2d direction;
    if (closest_idx < frenet_ref_->size() - 1) {
      direction = frenet_ref_->getPoint(closest_idx + 1) - closest_point;
    } else if (closest_idx > 0) {
      direction = closest_point - frenet_ref_->getPoint(closest_idx - 1);
    } else {
      direction = Eigen::Vector2d(1.0, 0.0); // 默认方向
    }
    
    // 计算点到参考线的有符号距离
    direction.normalize();
    Eigen::Vector2d normal(-direction.y(), direction.x()); // 法向量
    double signed_dist = (point - closest_point).dot(normal);
    
    // 获取走廊宽度
    double corridor_width = frenet_ref_->getCorridorDistanceAtIndex(closest_idx);
    double half_width = corridor_width / 2.0;
    
    // 如果距离绝对值小于等于半宽，说明点在走廊内
    return std::abs(signed_dist) <= half_width;
  }
  
  /**
   * @brief 获取点到走廊边界的距离
   * @param point 要判断的点（世界坐标系）
   * @return 距离值（正值表示在走廊内，负值表示超出走廊边界）
   */
  double getDistanceToCorridor(const Eigen::Vector2d &point) const {
    // 如果没有Frenet参考轨迹，返回一个很大的值（表示无限远）
    if (!frenet_ref_ || frenet_ref_->size() == 0) {
      return std::numeric_limits<double>::max();
    }
    
    // 找到最近的点
    Eigen::Vector2d closest_point;
    double min_dist = std::numeric_limits<double>::max();
    size_t closest_idx = 0;
    
    for (size_t i = 0; i < frenet_ref_->size(); ++i) {
      Eigen::Vector2d ref_point = frenet_ref_->getPoint(i);
      double dist = (point - ref_point).norm();
      if (dist < min_dist) {
        min_dist = dist;
        closest_point = ref_point;
        closest_idx = i;
      }
    }
    
    // 计算方向向量
    Eigen::Vector2d direction;
    if (closest_idx < frenet_ref_->size() - 1) {
      direction = frenet_ref_->getPoint(closest_idx + 1) - closest_point;
    } else if (closest_idx > 0) {
      direction = closest_point - frenet_ref_->getPoint(closest_idx - 1);
    } else {
      direction = Eigen::Vector2d(1.0, 0.0); // 默认方向
    }
    
    // 计算点到参考线的有符号距离
    direction.normalize();
    Eigen::Vector2d normal(-direction.y(), direction.x()); // 法向量
    double signed_dist = (point - closest_point).dot(normal);
    
    // 获取走廊宽度
    double corridor_width = frenet_ref_->getCorridorDistanceAtIndex(closest_idx);
    double half_width = corridor_width / 2.0;
    
    // 返回距离走廊边界的距离（正值表示在走廊内，负值表示超出走廊边界）
    return half_width - std::abs(signed_dist);
  }

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  boost::shared_ptr<FrenetReference> frenet_ref_;
};

} // namespace teb_local_planner

#endif // EDGE_FRENTE_CORRIDOR_H