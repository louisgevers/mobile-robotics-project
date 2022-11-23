from src import model
import cv2


def analyze_scene() -> model.World:
    # TODO dummy data
    return model.World(
        robot=model.Robot(
            position=model.Point(0, 0),
            angle=0,
        ),
        goal=model.Point(0, 0),
        obstacles=[],
    )


def get_webcam_capture(builtin: bool) -> cv2.VideoCapture:
    index = 0 if builtin else 2
    return cv2.VideoCapture(index)


def main():

    source = get_webcam_capture(builtin=False)

    while True:

        # Capture the video frame
        # by frame
        ret, frame = source.read()

        # Display the resulting frame
        cv2.imshow("frame", frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
