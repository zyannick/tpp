#pragma once

#include <iostream>

using namespace std; 

namespace tpp {
    
    enum class  TYPE_OF_FEATURES
    {
        PHASE_CONGRUENCY,
        ORB,
        FAST,
        SHI_TOMASI,
        SURF,
        AGAST,
        GFTT,
        HARRIS_CORNERS,
        AKAZE,
        BRISK,
        KAZE
    };
    
    TYPE_OF_FEATURES get_type_of_features_from_int(int value)
    {
        if(value == 0)
        {
            return TYPE_OF_FEATURES::PHASE_CONGRUENCY;
        }
        else if(value == 1)
        {
            return TYPE_OF_FEATURES::ORB;
        }
        else if(value == 2)
        {
            return TYPE_OF_FEATURES::FAST;
        }
        else if(value == 3)
        {
            return TYPE_OF_FEATURES::SHI_TOMASI;
        }
        else if(value == 4)
        {
            return TYPE_OF_FEATURES::SURF;
        }
        else if(value == 5)
        {
            return TYPE_OF_FEATURES::AGAST;
        }
        else if(value == 6)
        {
            return TYPE_OF_FEATURES::GFTT;
        }
        else if(value == 7)
        {
            return TYPE_OF_FEATURES::HARRIS_CORNERS;
        }
        else if(value == 9)
        {
            return TYPE_OF_FEATURES::AKAZE;
        }
        else if(value == 10)
        {
            return TYPE_OF_FEATURES::BRISK;
        }
        else if(value == 11)
        {
            return TYPE_OF_FEATURES::KAZE;
        }
    }
    
    
    
    string get_string_from_tof_enum(TYPE_OF_FEATURES tof)
    {
        if(TYPE_OF_FEATURES::PHASE_CONGRUENCY == tof)
        {
            return "phase_congruency";
        }
        else if(TYPE_OF_FEATURES::ORB == tof)
        {
            return "orb";
        }
        else if(TYPE_OF_FEATURES::FAST == tof)
        {
            return "fast";
        }
        else if(TYPE_OF_FEATURES::SHI_TOMASI == tof)
        {
            return "shi_thomasi";
        }
        else if(TYPE_OF_FEATURES::SURF == tof)
        {
            return "surf";
        }
        else if(TYPE_OF_FEATURES::AGAST == tof)
        {
            return "agast";
        }
        else if(TYPE_OF_FEATURES::GFTT == tof)
        {
            return "gftt";
        }
        else if(TYPE_OF_FEATURES::HARRIS_CORNERS == tof)
        {
            return "harris_corners";
        }
        else if(TYPE_OF_FEATURES::AKAZE == tof)
        {
            return "akaze";
        }
        else if(TYPE_OF_FEATURES::BRISK == tof)
        {
            return "brisk";
        }
        else if(TYPE_OF_FEATURES::KAZE == tof)
        {
            return "kaze";
        }
    }
    
    
    
}
