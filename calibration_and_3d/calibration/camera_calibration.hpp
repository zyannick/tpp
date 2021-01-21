#pragma once

#include "camera_calibration.hh"
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "algorithms/miscellaneous.hh"


namespace tpp
{
	stereo_params_eigen::stereo_params_eigen()
	{
		D1.resize(14,1);
        D1 << -3.5983545368377856e-01, 7.2718594775885714e-01, 0., 0., 0.,
                0., 0., 2.7842791627971475e+00, 0., 0., 0., 0., 0., 0. ;
		M1.resize(3, 3);
        M1 << 9.0731833340927309e+01, 0., 3.7742768390041384e+01, 0.,
                9.0729564315260347e+01, 2.8275954406100453e+01, 0., 0., 1. ;

		R1.resize(3, 3);
        R1 << 9.9964525627844780e-01, -9.5853277892722268e-03,
                2.4849207053108599e-02, 9.7240756879186600e-03,
                9.9993776545421076e-01, -5.4687814415572533e-03,
                -2.4795240511269186e-02, 5.7084769958455194e-03,
                9.9967625226288948e-01;
		P1.resize(3, 4);
        P1 << 3.4957869893593347e+01, 0., 2.6256224632263184e+01, 0., 0.,
                3.4957869893593347e+01, 2.2512099027633667e+01, 0., 0., 0., 1.,
                0.;

		D2.resize(14,1);
        D2 << -3.6132066363695586e-01, -1.3503587394586286e-01, 0., 0., 0.,
                0., 0., -1.4450109384070291e+00, 0., 0., 0., 0., 0., 0. ;

		M2.resize(3, 3);
        M2 << 9.0731833340927309e+01, 0., 3.9237413494295240e+01, 0.,
                9.0729564315260347e+01, 2.8464023868756911e+01, 0., 0., 1. ;

		R2.resize(3, 3);
        R2 << 9.9913922770807584e-01, 3.5761349755693034e-02,
                2.1022119744698746e-02, -3.5878311494126146e-02,
                9.9934256934728871e-01, 5.2130465842920443e-03,
                -2.0821873576600005e-02, -5.9627974987028895e-03,
                9.9976541979943956e-01;

		P2.resize(3, 4);
        P2 << 3.4957869893593347e+01, 0., 2.6256224632263184e+01,
                -4.9052225834509100e+03, 0., 3.4957869893593347e+01,
                2.2512099027633667e+01, 0., 0., 0., 1., 0. ;

		R.resize(3, 3);
        R << 9.9895218928679352e-01, -4.5572016814733322e-02,
                4.2088956461074903e-03, 4.5614195421987504e-02,
                9.9890355296444999e-01, -1.0537411968933059e-02,
                -3.7240696995213813e-03, 1.0718356144294842e-02,
                9.9993562200095520e-01;
		rotation_vector.resize(3,1);
		rotation_vector << 2.0911536031000683e-01, -8.9634705443727880e-01,
			1.0411203730150429e+00;

		T.resize(3,1);
        T << -1.4019733807246507e+02, -5.0179653671764770e+00,
                -2.9497843214584623e+00;

		F.resize(3, 3);
        F <<  4.5168937922347732e-06, 8.5269529063569674e-05,
                -1.6083963991871664e-02, -1.0224913340907528e-04,
                4.8258078187092561e-05, 3.7739306434009867e-01,
                -9.6360863686518095e-04, -3.7987563963884996e-01, 1.;

		E.resize(3, 3);
        E << 1.5323929126887070e-01, 2.8927656992585526e+00,
                -5.0487254132215700e+00, -3.4687981645140118e+00,
                1.6371126206401290e+00, 1.4017589711398099e+02,
                -1.3822812871736199e+00, -1.4027229791883224e+02,
                1.4984372008035751e+00;

		Q.resize(4, 4);
        Q <<  1., 0., 0., -2.6256224632263184e+01, 0., 1., 0.,
                -2.2512099027633667e+01, 0., 0., 0., 3.4957869893593347e+01, 0.,
                0., 7.1266633264580369e-03, 0.;
        //cout << "stereo_camera_calibration" << endl;
	}

    stereo_params_cv::stereo_params_cv()
    {

    }

	stereo_params_cv::stereo_params_cv(stereo_params_eigen st)
	{
        //std::cout << "dist eigen " << st.D1 << endl;
		D1 = eigen_to_mat_float(st.D1);
        //std::cout << "dist " << D1 << endl;
		M1 = eigen_to_mat_float(st.M1);
		R1 = eigen_to_mat_float(st.R1);
		P1 = eigen_to_mat_float(st.P1);

		D2 = eigen_to_mat_float(st.D2);
		M2 = eigen_to_mat_float(st.M2);
		R2 = eigen_to_mat_float(st.R2);
		P2 = eigen_to_mat_float(st.P2);

		R = eigen_to_mat_float(st.R);
		rotation_vector = eigen_to_mat_float(st.rotation_vector);
		T = eigen_to_mat_float(st.T);
		F = eigen_to_mat_float(st.F);
		E = eigen_to_mat_float(st.E);
		Q = eigen_to_mat_float(st.Q);
	}

	stereo_params_eigen init_stereo_calibration()
	{
		return stereo_params_eigen();
	}

    void stereo_params_cv::retreive_values()
    {
        std::string file_ex = std::string("");
        file_ex = file_ex.append(std::string("extrinsics").append(".yml"));

        cv::FileStorage fs;
        fs.open(file_ex, cv::FileStorage::READ);

        if (!fs.isOpened())
        {
            cerr << "Failed to open " << file_ex << endl;
            return;
        }

        fs["R"] >> R;
        fs["T"] >> T;
        fs["R1"] >> R1;
        fs["R2"] >> R2;
        fs["P1"] >> P1;
        fs["P2"] >> P2;
        fs["Q"] >> Q;
        fs["F"] >> F;
        fs["E"] >> E;

        fs.release();

        std::string file_in = std::string("");
        file_in = file_in.append(std::string("intrinsics").append(".yml"));

        fs.open(file_in, cv::FileStorage::READ);

        if (!fs.isOpened())
        {
            cerr << "Failed to open " << file_in << endl;
            return;
        }

        fs["M1"] >> M1;
        //cout << "taille "  << M1.size() << endl;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        fs.release();

    }
}
