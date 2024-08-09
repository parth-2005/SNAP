# SNAP - Real-time Face Filter App

**SNAP** is a real-time face filter application that uses MediaPipe and OpenCV to overlay filters on a live video feed. The current implementation includes a sunglasses filter, which can be applied to the detected face in the video stream.

## Features

- Real-time face detection using MediaPipe.
- Apply overlay filters to the detected face.
- Currently supports a sunglasses filter.

## Prerequisites

Make sure you have the following packages installed:
- `opencv-python`
- `mediapipe`
- `streamlit`
- `numpy`

You can install these packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone or download the repository.
2. Place your `sunglasses.png` filter image in the same directory as the script.
3. Run the application using Streamlit:

   ```bash
   streamlit run app.py
   ```

4. Open the Streamlit app in your browser (typically at `http://localhost:8501`).

5. Choose the "Sunglasses" filter from the sidebar to apply it to the live video feed.

## Demo

Here is a demo video of the application:

<video width="640" height="480" controls>
  <source src="demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
