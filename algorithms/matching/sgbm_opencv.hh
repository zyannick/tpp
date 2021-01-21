#pragma once


#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <string.h>
 
using namespace std;


namespace tpp {
    int apply_sgbm(Mat img1, Mat img2, Mat disparity)
    {
        Mat g1, g2;
        Mat disp, disp8;
        
        //char* method = argv[3];
        char* method = "SGBM";
        
        enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3, STEREO_3WAY=4 };
        int alg = STEREO_SGBM;
        int SADWindowSize, numberOfDisparities;
        bool no_display;
        float scale;
        
        /*
        
        Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
        
        if (!(strcmp(method, "BM")))
        {
            Ptr<StereoBM> sbm = StereoBM::create(16,9);
            sbm.state->SADWindowSize = 9;
            sbm.state->numberOfDisparities = 112;
            sbm.state->preFilterSize = 5;
            sbm.state->preFilterCap = 61;
            sbm.state->minDisparity = -39;
            sbm.state->textureThreshold = 507;
            sbm.state->uniquenessRatio = 0;
            sbm.state->speckleWindowSize = 0;
            sbm.state->speckleRange = 8;
            sbm.state->disp12MaxDiff = 1;
            sbm(g1, g2, disp);
        }
        else if (!(strcmp(method, "SGBM")))
        {
            StereoSGBM sbm;
            sbm.SADWindowSize = 3;
            sbm.numberOfDisparities = 144;
            sbm.preFilterCap = 63;
            sbm.minDisparity = -39;
            sbm.uniquenessRatio = 10;
            sbm.speckleWindowSize = 100;
            sbm.speckleRange = 32;
            sbm.disp12MaxDiff = 1;
            sbm.fullDP = false;
            sbm.P1 = 216;
            sbm.P2 = 864;
            sbm(g1, g2, disp);
        }
        */
        
        //normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
        
        disparity = disp8;
    }
}

