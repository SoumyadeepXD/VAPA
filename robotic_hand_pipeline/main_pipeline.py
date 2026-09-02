
import cv2

import config
from camera.realsense_capture import RealSenseCamera
from detection.detector_inference import ObjectDetector, draw_detections
from grasp.grasp_estimator import estimate_grasp
from force.force_lookup import estimate_force_n
from control.hand_controller import HandInterface, GraspController


def main():
    cam = RealSenseCamera()
    detector = ObjectDetector()
    hand = HandInterface(port=None)  # TODO: set your real serial/CAN port
    controller = GraspController(hand)

    print("Press 'g' to attempt a grasp on the best detection, "
          "'r' to release, 'q' to quit.")

    try:
        while True:
            color, depth_m = cam.get_frames()
            if color is None:
                continue

            detections = detector.detect(color)
            vis = draw_detections(color, detections)
            cv2.imshow("Pipeline", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elif key == ord("g") and detections:
                best = max(detections, key=lambda d: d.score)
                target = estimate_grasp(best, color, depth_m, cam)
                if target is None:
                    print(f"No valid depth for '{best.label}', skipping.")
                    continue

                force_n = estimate_force_n(target.label, target.width_m)
                print(f"Grasping '{target.label}': "
                      f"center={target.center_3d_m}, width={target.width_m:.3f}m, "
                      f"target_force={force_n:.2f}N")

                success = controller.execute_grasp(
                    target_width_m=target.width_m, target_force_n=force_n
                )
                print("Grasp converged." if success else "Grasp aborted (safety check).")

            elif key == ord("r"):
                controller.release()
                print("Released.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()