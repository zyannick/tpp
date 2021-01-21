#pragma once
#include <string.h>
#include <cctype>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <set>
#include <list>
#include <vector>

//#include <algorithms\miscellaneous\dirent.h>


using namespace std;

namespace tpp
{
	std::string ExePath() {
        //char buffer[MAX_PATH];
		//	GetModuleFileName(NULL, buffer, MAX_PATH);
        //string::size_type pos = std::string(buffer).find_last_of("\\/");
        //return std::string(buffer).substr(0, pos);
		return "";
	}

    void getListFilesOfDirectory(std::vector<std::string> &imageListCam1, std::vector<string>& imageListCam2, string dir_images = "")
	{
		string dir_cam1, dir_cam2;
		if (dir_images.length() == 0)
		{
			dir_images = ExePath();
			dir_images = dir_images.append("Images/");
		}
		dir_cam1 = dir_images.append("Camera1/");
		dir_cam2 = dir_images.append("Camera2/");

		char *c1 = new char[dir_cam1.length() + 1];
		//strcpy_s(c1, dir_cam1.c_str());

		char *c2 = new char[dir_cam2.length() + 1];
		//strcpy_s(c2, dir_cam2.c_str());
        /*
		DIR *dir;
		struct dirent *ent;
		if ((dir = opendir(c1)) != NULL) {
			while ((ent = readdir(dir)) != NULL) {
				imageListCam1.push_back(ent->d_name);
			}
			closedir(dir);
		}
		else {
			perror("");
			return;
		}

		if ((dir = opendir(c2)) != NULL) {
			while ((ent = readdir(dir)) != NULL) {
				imageListCam1.push_back(ent->d_name);
			}
			closedir(dir);
		}
		else {
			perror("");
			return;
        }*/
	}
}
