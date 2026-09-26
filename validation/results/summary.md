Mean Dice per class over the series (1 = perfect).

### baselines

| setting | method | Air | Aluminium | Electrolyte | Lithium | Product | mean | seconds | unclassified | smoothing |
|---|---|---|---|---|---|---|---|---|---|---|
| default | BiTS (locked, auto smoothing) | 0.975 | 0.941 | 0.994 | 0.983 | 0.788 | 0.936 | 4.24 | 0.038 | 8.000 |
| default | Otsu (neutron, 2 classes) | 0.825 | 0.000 | 0.876 | 0.000 | 0.167 | 0.373 | 0.02 | 0.000 |  |
| default | Multi-Otsu (neutron) | 0.978 | 0.919 | 0.775 | 0.922 | 0.254 | 0.769 | 21.7 | 0.000 |  |
| default | Multi-Otsu (X-ray) | 0.898 | 0.829 | 0.769 | 0.355 | 0.259 | 0.622 | 21.52 | 0.000 |  |
| default | K-means per timepoint* | 0.966 | 0.958 | 0.977 | 0.975 | 0.315 | 0.838 | 17.76 | 0.000 |  |
| default | GMM per timepoint* | 0.937 | 0.855 | 0.975 | 0.587 | 0.234 | 0.718 | 9.8 | 0.000 |  |
| default | Random walker | 0.999 | 0.994 | 0.995 | 0.866 | 0.506 | 0.872 | 13.55 | 0.000 |  |
| default | Pixel classifier (RF, ilastik features) | 0.985 | 0.932 | 0.969 | 0.915 | 0.726 | 0.905 | 1.12 | 0.000 |  |

### seeds

| setting | method | Air | Aluminium | Electrolyte | Lithium | Product | mean | seconds | unclassified |
|---|---|---|---|---|---|---|---|---|---|
| seed=0 | BiTS (locked, auto smoothing) | 0.975 | 0.941 | 0.994 | 0.983 | 0.788 | 0.936 | 4.09 | 0.038 |
| seed=0 | Random walker | 0.999 | 0.994 | 0.995 | 0.866 | 0.506 | 0.872 | 12.49 | 0.000 |
| seed=0 | Pixel classifier (RF, ilastik features) | 0.985 | 0.932 | 0.969 | 0.915 | 0.726 | 0.905 | 1.03 | 0.000 |
| seed=1 | BiTS (locked, auto smoothing) | 0.976 | 0.945 | 0.994 | 0.983 | 0.781 | 0.936 | 4.12 | 0.036 |
| seed=1 | Random walker | 0.999 | 0.994 | 0.995 | 0.867 | 0.505 | 0.872 | 14.55 | 0.000 |
| seed=1 | Pixel classifier (RF, ilastik features) | 0.985 | 0.929 | 0.968 | 0.894 | 0.753 | 0.906 | 1.01 | 0.000 |
| seed=2 | BiTS (locked, auto smoothing) | 0.969 | 0.925 | 0.986 | 0.976 | 0.743 | 0.919 | 2.78 | 0.053 |
| seed=2 | Random walker | 0.999 | 0.994 | 0.995 | 0.866 | 0.505 | 0.872 | 14.11 | 0.000 |
| seed=2 | Pixel classifier (RF, ilastik features) | 0.985 | 0.934 | 0.969 | 0.911 | 0.735 | 0.907 | 1.52 | 0.000 |
| seed=3 | BiTS (locked, auto smoothing) | 0.976 | 0.941 | 0.994 | 0.983 | 0.773 | 0.933 | 4.46 | 0.037 |
| seed=3 | Random walker | 0.999 | 0.994 | 0.995 | 0.868 | 0.504 | 0.872 | 9.8 | 0.000 |
| seed=3 | Pixel classifier (RF, ilastik features) | 0.985 | 0.932 | 0.969 | 0.909 | 0.720 | 0.903 | 1.29 | 0.000 |
| seed=4 | BiTS (locked, auto smoothing) | 0.976 | 0.940 | 0.994 | 0.983 | 0.791 | 0.937 | 3.98 | 0.038 |
| seed=4 | Random walker | 0.999 | 0.994 | 0.995 | 0.865 | 0.505 | 0.872 | 10.29 | 0.000 |
| seed=4 | Pixel classifier (RF, ilastik features) | 0.986 | 0.938 | 0.970 | 0.900 | 0.741 | 0.907 | 1.08 | 0.000 |
| mean ± sd | BiTS (locked, auto smoothing) |  |  |  |  | 0.775 ± 0.017 | 0.932 ± 0.007 |  |  |
| mean ± sd | Random walker |  |  |  |  | 0.505 ± 0.001 | 0.872 ± 0.000 |  |  |
| mean ± sd | Pixel classifier (RF, ilastik features) |  |  |  |  | 0.735 ± 0.012 | 0.906 ± 0.002 |  |  |

### noise

| setting | method | Air | Aluminium | Electrolyte | Lithium | Product | mean | seconds | unclassified |
|---|---|---|---|---|---|---|---|---|---|
| sigma=40 | BiTS (locked, auto smoothing) | 0.946 | 0.841 | 0.973 | 0.953 | 0.719 | 0.886 | 2.12 | 0.108 |
| sigma=40 | Random walker | 1.000 | 0.999 | 0.996 | 0.865 | 0.508 | 0.874 | 17.83 | 0.000 |
| sigma=40 | Pixel classifier (RF, ilastik features) | 0.996 | 0.975 | 0.976 | 0.899 | 0.750 | 0.919 | 1.0 | 0.000 |
| sigma=60 | BiTS (locked, auto smoothing) | 0.975 | 0.941 | 0.994 | 0.983 | 0.788 | 0.936 | 4.02 | 0.038 |
| sigma=60 | Random walker | 0.999 | 0.994 | 0.995 | 0.866 | 0.506 | 0.872 | 12.34 | 0.000 |
| sigma=60 | Pixel classifier (RF, ilastik features) | 0.985 | 0.932 | 0.969 | 0.915 | 0.726 | 0.905 | 0.98 | 0.000 |
| sigma=90 | BiTS (locked, auto smoothing) | 0.991 | 0.976 | 0.984 | 0.971 | 0.653 | 0.915 | 3.29 | 0.009 |
| sigma=90 | Random walker | 0.996 | 0.979 | 0.991 | 0.864 | 0.497 | 0.865 | 14.51 | 0.000 |
| sigma=90 | Pixel classifier (RF, ilastik features) | 0.983 | 0.931 | 0.970 | 0.890 | 0.740 | 0.903 | 1.02 | 0.000 |
| sigma=120 | BiTS (locked, auto smoothing) | 0.992 | 0.968 | 0.960 | 0.933 | 0.488 | 0.868 | 1.78 | 0.001 |
| sigma=120 | Random walker | 0.992 | 0.962 | 0.986 | 0.852 | 0.486 | 0.856 | 4.08 | 0.000 |
| sigma=120 | Pixel classifier (RF, ilastik features) | 0.984 | 0.893 | 0.956 | 0.890 | 0.734 | 0.892 | 0.92 | 0.000 |

### smoothing

| setting | method | Air | Aluminium | Electrolyte | Lithium | Product | mean | seconds | unclassified | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| sigma=60 | beta=0 | 0.968 | 0.923 | 0.986 | 0.978 | 0.753 | 0.922 | 0.05 | 0.053 |  |
| sigma=60 | beta=0.5 | 0.971 | 0.930 | 0.989 | 0.981 | 0.775 | 0.929 | 0.96 | 0.048 |  |
| sigma=60 | beta=1 | 0.972 | 0.934 | 0.990 | 0.982 | 0.781 | 0.932 | 0.98 | 0.045 |  |
| sigma=60 | beta=2 | 0.973 | 0.937 | 0.992 | 0.982 | 0.787 | 0.934 | 0.95 | 0.042 |  |
| sigma=60 | beta=4 | 0.975 | 0.940 | 0.993 | 0.983 | 0.789 | 0.936 | 0.91 | 0.040 |  |
| sigma=60 | beta=8 | 0.975 | 0.941 | 0.994 | 0.983 | 0.788 | 0.936 | 0.91 | 0.038 |  |
| sigma=60 | beta=16 | 0.976 | 0.943 | 0.995 | 0.983 | 0.788 | 0.937 | 0.88 | 0.037 |  |
| sigma=60 | beta=32 | 0.976 | 0.944 | 0.995 | 0.984 | 0.789 | 0.937 | 0.88 | 0.036 |  |
| sigma=60 | auto (beta=8) | 0.975 | 0.941 | 0.994 | 0.983 | 0.788 | 0.936 | 4.13 | 0.038 | labels settled: the next setting changes 0.15% of voxels |
| sigma=90 | beta=0 | 0.990 | 0.972 | 0.982 | 0.966 | 0.635 | 0.909 | 0.05 | 0.011 |  |
| sigma=90 | beta=0.5 | 0.992 | 0.978 | 0.986 | 0.974 | 0.667 | 0.919 | 0.82 | 0.009 |  |
| sigma=90 | beta=1 | 0.993 | 0.981 | 0.988 | 0.977 | 0.681 | 0.924 | 0.83 | 0.007 |  |
| sigma=90 | beta=2 | 0.995 | 0.983 | 0.990 | 0.979 | 0.692 | 0.928 | 0.84 | 0.006 |  |
| sigma=90 | beta=4 | 0.995 | 0.986 | 0.991 | 0.981 | 0.694 | 0.929 | 0.91 | 0.005 |  |
| sigma=90 | beta=8 | 0.996 | 0.986 | 0.992 | 0.982 | 0.695 | 0.930 | 0.87 | 0.004 |  |
| sigma=90 | beta=16 | 0.996 | 0.987 | 0.992 | 0.982 | 0.698 | 0.931 | 0.8 | 0.003 |  |
| sigma=90 | beta=32 | 0.996 | 0.986 | 0.992 | 0.983 | 0.701 | 0.932 | 0.79 | 0.003 |  |
| sigma=90 | auto (beta=0.25) | 0.991 | 0.976 | 0.984 | 0.971 | 0.653 | 0.915 | 3.3 | 0.009 | strongest setting before a guard failed at 0.5 |

### class_birth

| setting | method | Air | Aluminium | Electrolyte | Lithium | Product | mean | seconds | unclassified |
|---|---|---|---|---|---|---|---|---|---|
| product absent at T0 | all classes from T0 | 0.968 | 0.923 | 0.977 | 0.978 | 0.167 | 0.803 | 2.42 | 0.076 |
| product absent at T0 | defined where it exists | 0.975 | 0.941 | 0.994 | 0.983 | 0.788 | 0.936 | 3.99 | 0.038 |

### drift

| setting | method | Air | Aluminium | Electrolyte | Lithium | Product | mean | seconds | worst_fixed_class_dice |
|---|---|---|---|---|---|---|---|---|---|
| gain +0%/step | BiTS locked | 0.976 | 0.944 | 0.995 | 0.944 | 0.838 | 0.939 | 6.01 | 0.673 |
| gain +0%/step | adaptive, frozen boundaries | 0.939 | 0.873 | 0.967 | 0.665 | 0.125 | 0.714 | 2.12 | 0.000 |
| gain +0%/step | adaptive, anchored drift | 0.939 | 0.923 | 0.968 | 0.710 | 0.125 | 0.733 | 1.63 | 0.000 |
| gain +3%/step | BiTS locked | 0.954 | 0.944 | 0.970 | 0.834 | 0.767 | 0.894 | 3.71 | 0.048 |
| gain +3%/step | adaptive, frozen boundaries | 0.936 | 0.947 | 0.967 | 0.624 | 0.125 | 0.720 | 1.39 | 0.000 |
| gain +3%/step | adaptive, anchored drift | 0.938 | 0.952 | 0.967 | 0.730 | 0.125 | 0.742 | 1.3 | 0.000 |
| gain +6%/step | BiTS locked | 0.946 | 0.760 | 0.906 | 0.740 | 0.609 | 0.792 | 3.6 | 0.000 |
| gain +6%/step | adaptive, frozen boundaries | 0.936 | 0.969 | 0.970 | 0.575 | 0.125 | 0.715 | 1.36 | 0.000 |
| gain +6%/step | adaptive, anchored drift | 0.936 | 0.972 | 0.967 | 0.694 | 0.125 | 0.739 | 1.12 | 0.000 |

### misregistration

| setting | method | Air | Aluminium | Electrolyte | Lithium | Product | mean | seconds | unclassified | estimated_shift | applied_shift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| offset (0.0, 0.0, 0.0) | uncorrected | 0.975 | 0.941 | 0.994 | 0.983 | 0.788 | 0.936 | 4.01 | 0.038 |  |  |
| offset (0.0, 0.0, 0.0) | after Check Alignment | 0.975 | 0.941 | 0.994 | 0.983 | 0.788 | 0.936 | 4.02 | 0.038 | +0.00 +0.01 +0.00 | +0 +0 +0 |
| offset (0.0, 0.5, 0.5) | uncorrected | 0.970 | 0.931 | 0.980 | 0.970 | 0.703 | 0.911 | 3.14 | 0.028 |  |  |
| offset (0.0, 0.5, 0.5) | after Check Alignment | 0.970 | 0.931 | 0.980 | 0.970 | 0.703 | 0.911 | 3.19 | 0.028 | +0.00 -0.44 -0.60 | +0 +0 +0 |
| offset (0.0, 1.0, 1.0) | uncorrected | 0.950 | 0.881 | 0.966 | 0.924 | 0.550 | 0.854 | 2.25 | 0.015 |  |  |
| offset (0.0, 1.0, 1.0) | after Check Alignment | 0.973 | 0.944 | 0.995 | 0.982 | 0.779 | 0.934 | 4.33 | 0.038 | +0.00 -0.97 -1.01 | +0 -1 -1 |
| offset (0.0, 2.0, 2.0) | uncorrected | 0.973 | 0.920 | 0.971 | 0.934 | 0.625 | 0.885 | 2.94 | 0.006 |  |  |
| offset (0.0, 2.0, 2.0) | after Check Alignment | 0.970 | 0.940 | 0.995 | 0.983 | 0.790 | 0.936 | 4.34 | 0.040 | +0.00 -1.98 -2.01 | +0 -2 -2 |
| offset (2.0, 0.0, 0.0) (closed ends) | uncorrected | 0.971 | 0.864 | 0.967 | 0.801 | 0.468 | 0.814 | 2.99 | 0.000 |  |  |
| offset (2.0, 0.0, 0.0) (closed ends) | after Check Alignment | 0.976 | 0.856 | 0.966 | 0.867 | 0.627 | 0.858 | 2.45 | 0.070 | -2.05 +0.01 -0.01 | -2 +0 +0 |

### class_shapes

| setting | method | Air | Aluminium | Electrolyte | Lithium | Product | mean | seconds | unclassified | components |
|---|---|---|---|---|---|---|---|---|---|---|
| two-state lithium | one Gaussian | 0.970 | 0.931 | 0.987 | 0.938 | 0.695 | 0.904 | 3.76 | 0.049 | Air:1 Aluminium:1 Electrolyte:1 Lithium:1 Product:1 |
| two-state lithium | mixture (BIC, up to 3) | 0.970 | 0.931 | 0.987 | 0.953 | 0.729 | 0.914 | 10.24 | 0.049 | Air:1 Aluminium:1 Electrolyte:1 Lithium:3 Product:1 |
| cupping 0.3 | one Gaussian | 0.983 | 0.960 | 0.993 | 0.977 | 0.763 | 0.935 | 3.58 | 0.027 | Air:1 Aluminium:1 Electrolyte:1 Lithium:1 Product:1 |
| cupping 0.3 | mixture (BIC, up to 3) | 0.983 | 0.960 | 0.993 | 0.977 | 0.763 | 0.935 | 8.98 | 0.027 | Air:1 Aluminium:1 Electrolyte:1 Lithium:1 Product:1 |

### coefficients

| setting | method | Air | Aluminium | Electrolyte | Lithium | Product | mean | seconds | unclassified |
|---|---|---|---|---|---|---|---|---|---|
| Product | region (defined where it exists) | 0.975 | 0.941 | 0.994 | 0.983 | 0.788 | 0.936 | 3.96 | 0.038 |
| Product | no definition (from T0 only) | 0.968 | 0.923 | 0.977 | 0.978 | 0.167 | 0.803 | 2.49 | 0.076 |
| Product | coefficients, exact | 0.970 | 0.928 | 0.987 | 0.979 | 0.752 | 0.923 | 3.69 | 0.049 |
| Product | coefficients, +5% error | 0.975 | 0.942 | 0.995 | 0.982 | 0.793 | 0.937 | 3.93 | 0.037 |
| Product | coefficients, +15% error | 0.968 | 0.923 | 0.980 | 0.978 | 0.647 | 0.899 | 2.85 | 0.054 |
