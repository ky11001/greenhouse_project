import open3d as o3d


class Open3dICP:

    def __init__(self,
                 error_metric: str = 'point-to-point',
                 max_correspondence_distance: float = 0.01,
                 max_iteration: int = 1000):
        assert error_metric in ['point-to-point', 'point-to-plane', 'generalised-icp']

        self.error_metric = error_metric
        self.max_correspondence_distance = max_correspondence_distance

        if error_metric == 'point-to-point':
            self.estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        elif error_metric == 'point-to-plane':
            self.estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        elif error_metric == 'generalised-icp':
            self.estimation_method = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()

        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)

    def estimate_relative_pose(self, o3d_source_pcd, o3d_target_pcd, T_init):

        result = o3d.pipelines.registration.registration_icp(
            source=o3d_source_pcd,
            target=o3d_target_pcd,
            max_correspondence_distance=self.max_correspondence_distance,
            init=T_init,
            estimation_method=self.estimation_method,
            criteria=self.criteria)

        return result.transformation, result.fitness, result.inlier_rmse, len(result.correspondence_set)
