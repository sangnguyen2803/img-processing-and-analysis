# Video Processing Lab: Tennis Shot Analysis

**Student:** NGUYEN Sang  
**Lab:** Video Processing



## Dependencies

- `opencv-python` (cv2): Video processing and computer vision
- `numpy`: Numerical computations
- `pandas`: Data manipulation and CSV handling
- `matplotlib`: Visualization and plotting


## Project Structure

```
VideoProcessingLab/
├── README.md                                      # Project documentation
├── report.ipynb                                   # Main analysis notebook
├── video_files/
│   └── videoTEST02.mp4                           # Input video file
├── plots/
│   └── videoTEST02_meandiff.png                  # Shot boundary detection visualization
├── results/
│   ├── videoTEST02_shots_summary.csv             # All detected shots with features
│   ├── videoTEST02_filtered_shots.csv            # Filtered effective-play candidates
│   └── effective_play_ranked_approach_net.csv    # Final ranked approach-to-net shots
└── shots_output/
    └── shot_*.mp4                                 # Individual shot video clips
```
## Output Files

| File | Description |
|------|-------------|
| `videoTEST02_meandiff.png` | Visualization of shot boundaries with threshold line |
| `videoTEST02_shots_summary.csv` | All detected shots with frame numbers, timestamps, and features |
| `videoTEST02_filtered_shots.csv` | Effective-play shots after threshold filtering |
| `effective_play_ranked_approach_net.csv` | Final ranked shots with approach scores and motion features |
| `shots_output/shot_*.mp4` | Individual video clips for each detected shot |

## Key Functions

### Shot Detection
- `mean_abs_diff_between_frames()`: Computes pixel-level frame difference
- `detect_shot_boundaries()`: Main shot detection function
- `plot_meandiff_func()`: Visualizes shot boundaries

### Shot Analysis
- `estimate_court_hue()`: Estimates dominant court color
- `court_ratio_from_hue()`: Computes court pixel fraction
- `analyze_shot()`: Extracts per-shot features

### Motion Analysis
- `centroid_track_mog2()`: Tracks player centroid with background subtraction
- `features_maxadvance_and_center()`: Computes motion-based features
- `zscore()`: Normalizes feature vectors
- `sigmoid()`: Applies sigmoid transformation

### Utilities
- `try_open_writer()`: Handles video codec selection
- `save_shot_clip()`: Extracts and saves individual shot clips
- `summarize_and_save()`: Exports results to CSV

## Input Parameters

### Shot Detection
- `diff_threshold`: Mean absolute difference threshold for cut detection (default: 100.0)
- `min_shot_frames`: Minimum frame count for valid shots (default: 8)
- `max_frames`: Maximum frames to process, 0 for all (default: 0)

### Feature Extraction
- `hue_tol`: Tolerance for court hue matching in HSV (default: 12)
- `sample_step`: Frame sampling interval for feature computation (default: 2)

### Shot Filtering
- `court_ratio_threshold`: Minimum court presence ratio (default: 0.52)
- `meandiff_threshold`: Maximum motion intensity threshold (default: 10)

### Motion Analysis
- `roi`: Region of interest for centroid tracking
- `area_min`: Minimum blob area for valid centroid detection (default: 300)


