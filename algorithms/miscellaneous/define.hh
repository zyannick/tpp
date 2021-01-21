#ifndef DEFINE_HH
#define DEFINE_HH

enum class MODULE_SELECTION  { DISPARITY_MAP ,  STEREO, FEATURE_EXTRACTION, FEATURE_LABELLING, OPTICAL_FLOW, CONVERT_IMAGES , NOTHING };

MODULE_SELECTION from_int_module_selection(int i)
{
    if(i == 1)
    {
        return MODULE_SELECTION::DISPARITY_MAP;
    }
    else if (i == 2) {
        return MODULE_SELECTION::STEREO;
    }
    else if (i == 3) {
        return MODULE_SELECTION::FEATURE_EXTRACTION;
    }
    else if (i == 4) {
        return MODULE_SELECTION::FEATURE_LABELLING;
    }
    else if (i == 5) {
        return MODULE_SELECTION::OPTICAL_FLOW;
    }
    else if (i == 6) {
        return MODULE_SELECTION::CONVERT_IMAGES;
    }
    else if (i == 7) {
        return MODULE_SELECTION::NOTHING;
    }
}


enum class POINT_OR_SILHOUETTE { POINT , SILHOUETTE };

enum class TYPE_GRAPHISME  { BOX_PLOT ,  LINE_CHART , BAR_CHART  };

enum class ID_VALUE  { RMS ,  T_NORME , AVERAGE_EP , FOCAL_DISTANCE  };

enum class REP_TYPE  { INDIVIDUAL ,  TOGETHER };


#endif // DEFINE_HH
