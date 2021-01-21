#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <memory>


using namespace Eigen;
using namespace std;


namespace tpp
{
	struct stereo_match
	{
        std::vector<Vector2d> list_first_view_matches;
        std::vector<Vector2d> list_second_view_matches;

        Vector2d first_point;
        Vector2d second_point;
        Vector3d point_projected;
        int depth;
        int id = -1;
        double strength;
        double similarity = 0;
        double lowe_ratio = -1;
        int seen = -1;
        int object_number = 0;
        int numero_frame = 1;
        int taken_time_consistency = -1;
        int taken_left_right_consistency = -1;
        bool has_previous = false;
        int taken = -1;


        inline
        stereo_match()
        {
            first_point << 0, 0;
            second_point << 0, 0;
            similarity = 0;
            list_first_view_matches.push_back(first_point);
            list_second_view_matches.push_back(second_point);
            depth = list_first_view_matches.size();
        }



        inline
        stereo_match(Vector2d pl, Vector2d pr, double sim)
        {
            first_point = pl;
            second_point = pr;
            similarity = sim;
            strength = 0;
            list_first_view_matches.push_back(pl);
            list_second_view_matches.push_back(pr);
            depth = list_first_view_matches.size();
        }

        inline
        stereo_match(Vector2d pl, Vector2d pr, Vector3d projected)
        {
            first_point = pl;
            second_point = pr;
            point_projected = projected;
            strength = 0;
            list_first_view_matches.push_back(pl);
            list_second_view_matches.push_back(pr);
            depth = list_first_view_matches.size();
        }



        inline
        stereo_match(Vector2d pl, Vector2d pr, double sim, double st)
        {
            first_point = pl;
            second_point = pr;
            similarity = sim;
            strength = st;
            list_first_view_matches.push_back(pl);
            list_second_view_matches.push_back(pr);
            depth = list_first_view_matches.size();
        }


        inline
        stereo_match(Vector2d pl, Vector2d pr, double sim, double st, double d)
        {
            first_point = pl;
            second_point = pr;
            similarity = sim;
            strength = st;
            lowe_ratio = d;
            list_first_view_matches.push_back(pl);
            list_second_view_matches.push_back(pr);
            depth = list_first_view_matches.size();
        }

        inline
        stereo_match(Vector2d pl, Vector2d pr, stereo_match prev_match)
        {
            id = prev_match.id;
            object_number = prev_match.object_number;
            first_point = pl;
            second_point = pr;
            list_first_view_matches = prev_match.list_first_view_matches;
            list_second_view_matches = prev_match.list_second_view_matches;
            list_first_view_matches.push_back(pl);
            list_second_view_matches.push_back(pr);
            depth = list_first_view_matches.size();
            has_previous = true;
        }




        inline void initialize_taken_values()
        {
            taken_time_consistency = -1;
            taken_left_right_consistency = -1;
            taken = -1;
        }

        inline void update_match(stereo_match scb)
        {

        }

        inline
        void  update_match(Vector2d pl, Vector2d pr, double sim, stereo_match scb)
        {
            first_point = pl;
            second_point = pr;
            similarity = sim;
        }

        inline
        std::pair<double, double> operator -(stereo_match other)
        {
            return pair<double, double>( (other.first_point - first_point).norm(), (other.second_point - second_point).norm());
        }








	};

    inline
    ostream& operator <<(ostream& os, const stereo_match &st)
    {
        os << st.first_point.transpose() << "---" << st.second_point.transpose();
        return os;
    }


    /*struct left_right_matches
	{
		int sz;
		std::vector<stereo_match> list_st;

        left_right_matches()
        {

        }

        left_right_matches(int n)
        {
            sz = n;
        }

        void add_match(stereo_match sm)
        {
            if (list_st.size() == 0)
            {
                list_st.push_back(sm);
            }
            else
            {
                list_st.push_back(sm);
                std::sort(begin(list_st), end(list_st), decrescendo_similarity());
                if (list_st.size() > sz)
                    list_st.pop_back();
            }
        }
    };*/



	inline
        double compute_rho(Vector2i pt)
	{
        return sqrt(pt.x() * pt.x() + pt.y() * pt.y());
	}

	inline
        bool approx_number(double x, double y, double eps)
	{
		if (fabs(x - y) <= eps)
			return true;
		else
			return false;
	}



	struct higher_similarity_vect
	{
        inline bool operator() (const Vector3d& sim1, const Vector3d& sim2)
		{
            return (sim1.z() > sim2.z());
		}
	};



    /*struct decrescendo_similarity_list
	{
		inline bool operator() (const left_right_matches& match1, const left_right_matches& match2)
		{
			assert(match1.list_st.size() != 0 && match2.list_st.size() != 0);
			return (match1.list_st[0].similarity > match2.list_st[0].similarity);
		}
    };*/

	struct crescendo_similarity
	{
		inline bool operator() (const stereo_match& match1, const stereo_match& match2)
		{
			return (match1.similarity < match2.similarity);
		}
	};

	struct decrescendo_similarity
	{
		inline bool operator() (const stereo_match& match1, const stereo_match& match2)
		{
			return (match1.similarity > match2.similarity);
		}
	};

    struct crescendo_magnitude
    {
        inline bool operator() (const stereo_match& match1, const stereo_match& match2)
        {
            return (match1.strength < match2.strength);
        }
    };

    struct decrescendo_magnitude
    {
        inline bool operator() (const stereo_match& match1, const stereo_match& match2)
        {
            return (match1.strength > match2.strength);
        }
    };


	struct order_epipolar
	{
		inline bool operator()  (const stereo_match& match1, const stereo_match& match2)
		{
            return (match1.first_point.y() > match2.first_point.y());
		}
	};
    
    struct crescendo_norm
	{
		inline bool operator() (const stereo_match& match1, const stereo_match& match2)
		{
			return (match1.first_point.norm() < match2.first_point.norm());
		}
	};



}

