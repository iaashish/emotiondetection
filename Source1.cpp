#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv\cv.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <Windows.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <dirent.h>

using namespace std;
using namespace cv;

// Functions for facial feature detection

static void detectFaces(Mat&, vector<Rect_<int> >&, string);
static void detectEyes(Mat&, vector<Rect_<int> >&, string);
static void detectNose(Mat&, vector<Rect_<int> >&, string);
static void detectMouth(Mat&, vector<Rect_<int> >&, string);
static void detectFacialFeaures(Mat&, const vector<Rect_<int> >, string, string, string);
static float setsvm(String, String, String, String, String);

string input_image_path;
string face_cascade_path, eye_cascade_path, nose_cascade_path, mouth_cascade_path;

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv,
		"{eyes||}{nose||}{mouth||}");

	input_image_path = "C://irs/AF02HAS.jpg";
	face_cascade_path = "C://irs/haarcascade_frontalface_alt.xml";
	eye_cascade_path = "C://irs/haarcascade_eye.xml";
	nose_cascade_path = "C://irs/haarcascade_mcs_nose.xml";
	mouth_cascade_path = "C://irs/haarcascade_mcs_mouth.xml";

	String pImg = "F:\\CODES\\OPENCVCodes\\DatasetSVM\\disgust1.jpg";

	
	if (input_image_path.empty() || face_cascade_path.empty())
	{
		cout << "IMAGE or FACE_CASCADE are not specified";
		return 1;
	}
	// Load image and cascade classifier files
	Mat image;
	image = imread(input_image_path);

	// Detect faces and facial features
	vector<Rect_<int> > faces;
	wchar_t szFileName[MAX_PATH] = { 0 };
	OPENFILENAMEW ofn;
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrFile = szFileName;

	GetSaveFileNameW(&ofn);

	wstring ws(szFileName);
	string str(ws.begin(), ws.end());

	cv::Mat frame = imread(str);
	detectFaces(frame, faces, face_cascade_path);
	detectFacialFeaures(frame, faces, eye_cascade_path, nose_cascade_path, mouth_cascade_path);
	DIR *dir;
	struct dirent *ent;
	std::vector<std::string> collection1;
	if ((dir = opendir("c:\\irs\\KDEF")) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			collection1.push_back(ent->d_name);
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}
	String mood;
	int NESCount = 0;
	int HASCount = 0;
	int SASCount = 0;
	int AFSCount = 0;
	int DISCount = 0;

	//dirent end
	
		//cout << collection1[i] << endl;
	String timg1 = "C:\\irs\\IMDataset\\girl.jpg";
		String timg2 = "C:\\irs\\IMDataset\\smiling2.jpg";
		String timg3 = "C:\\irs\\IMDataset\\sad.jpg";
		String timg4 = "C:\\irs\\IMDataset\\AF01AFS.JPG";
		
		float res = setsvm(timg1, timg2, timg3, timg4, str);
		if (res == -1)
			mood =  "neutral";
		else if (res == 1)
			mood = "smiling";
		else if (res == 2)
			mood =  "sad";
		else if (res == 3)
			mood = "fear";
	putText(frame, mood, Point(50, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 200, 200), 4);
	imshow("Result", frame);

	waitKey(0);
	return 0;
}


static void detectFaces(Mat& img, vector<Rect_<int> >& faces, string cascade_path)
{
	CascadeClassifier face_cascade;
	face_cascade.load(cascade_path);

	face_cascade.detectMultiScale(img, faces, 1.15, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}

static void detectFacialFeaures(Mat& img, const vector<Rect_<int> > faces, string eye_cascade,
	string nose_cascade, string mouth_cascade)
{
	for (unsigned int i = 0; i < faces.size(); ++i)
	{
		// Mark the bounding box enclosing the face
		Rect face = faces[i];
		rectangle(img, Point(face.x, face.y), Point(face.x + face.width, face.y + face.height),
			Scalar(255, 0, 0), 1, 4);

		// Eyes, nose and mouth will be detected inside the face (region of interest)
		Mat ROI = img(Rect(face.x, face.y, face.width, face.height));

		// Check if all features (eyes, nose and mouth) are being detected
		bool is_full_detection = false;
		if ((!eye_cascade.empty()) && (!nose_cascade.empty()) && (!mouth_cascade.empty()))
			is_full_detection = true;

		// Detect eyes if classifier provided by the user
		if (!eye_cascade.empty())
		{
			vector<Rect_<int> > eyes;
			detectEyes(ROI, eyes, eye_cascade);

			// Mark points corresponding to the centre of the eyes
			for (unsigned int j = 0; j < eyes.size(); ++j)
			{
				Rect e = eyes[j];
				//circle(ROI, Point(e.x + e.width / 2, e.y + e.height / 2), 3, Scalar(0, 255, 0), -1, 8);
				rectangle(ROI, Point(e.x, e.y), Point(e.x + e.width, e.y + e.height),
					Scalar(0, 255, 0), 1, 4);
			}
		}

		// Detect nose if classifier provided by the user
		double nose_center_height = 0.0;
		if (!nose_cascade.empty())
		{
			vector<Rect_<int> > nose;
			detectNose(ROI, nose, nose_cascade);

			// Mark points corresponding to the centre (tip) of the nose
			for (unsigned int j = 0; j < nose.size(); ++j)
			{
				Rect n = nose[j];
				//circle(ROI, Point(n.x + n.width / 2, n.y + n.height / 2), 3, Scalar(0, 255, 0), -1, 8);
				nose_center_height = (n.y + n.height / 2);
			}
		}

		// Detect mouth if classifier provided by the user
		double mouth_center_height = 0.0;
		if (!mouth_cascade.empty())
		{
			vector<Rect_<int> > mouth;
			detectMouth(ROI, mouth, mouth_cascade);

			for (unsigned int j = 0; j < mouth.size(); ++j)
			{
				Rect m = mouth[j];
				mouth_center_height = (m.y + m.height / 2);

				// The mouth should lie below the nose
				if ((is_full_detection) && (mouth_center_height > nose_center_height))
				{
					rectangle(ROI, Point(m.x, m.y), Point(m.x + m.width, m.y + m.height), Scalar(0, 255, 0), 1, 4);
				}
				else if ((is_full_detection) && (mouth_center_height <= nose_center_height))
					continue;
				else
					rectangle(ROI, Point(m.x, m.y), Point(m.x + m.width, m.y + m.height), Scalar(0, 255, 0), 1, 4);
			}
		}

	}

	return;
}

static void detectEyes(Mat& img, vector<Rect_<int> >& eyes, string cascade_path)
{
	CascadeClassifier eyes_cascade;
	eyes_cascade.load(cascade_path);

	eyes_cascade.detectMultiScale(img, eyes, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}

static void detectNose(Mat& img, vector<Rect_<int> >& nose, string cascade_path)
{
	CascadeClassifier nose_cascade;
	nose_cascade.load(cascade_path);

	nose_cascade.detectMultiScale(img, nose, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}

static void detectMouth(Mat& img, vector<Rect_<int> >& mouth, string cascade_path)
{
	CascadeClassifier mouth_cascade;
	mouth_cascade.load(cascade_path);

	mouth_cascade.detectMultiScale(img, mouth, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}// another working code

float setsvm(String tImage1, String tImage2, String tImage3, String tImage4, String pImage){
	// Data for visual representation
	//dirent start

	
	int num_files = 4;
	int h = 128, w = 128;
	Size size(h, w);
	/*Setup for 'Neutral Face'*/
	// Set up training data 
	Mat image[4];
	image[0] = imread(tImage1, 0);
	image[1] = imread(tImage2, 0); 
	image[2] = imread(tImage3, 0);
	image[3] = imread(tImage4, 0);

	resize(image[0], image[0], Size(w, h));
	resize(image[1], image[1], Size(w, h));
	resize(image[2], image[2], Size(w, h));
	resize(image[3], image[3], Size(w, h));

	Mat trainingDataMat(4, h*w, CV_32FC1); //Training sample from input images
	int ii = 0;
	for (int i = 0; i < num_files; i++){
		Mat temp = image[i];
		ii = 0;
		for (int j = 0; j < temp.rows; j++){
			for (int k = 0; k < temp.cols; k++){
				trainingDataMat.at<float>(i, ii++) = temp.at<uchar>(j, k);
			}
		}
	}
	//Set up labels 
	Mat labels(num_files, 1, CV_32FC1);
	labels.at<float>(0, 0) = -1.0;  //neutral
	labels.at<float>(1, 0) = 1.0;   //smiling
	labels.at<float>(2, 0) = 2.0;   //sad
	labels.at<float>(3, 0) = 3.0;   //fear

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.gamma = 3;

	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train the SVM
	CvSVM svm;
	svm.train(trainingDataMat, labels, Mat(), Mat(), params);
	svm.save("svm.xml"); // saving
	svm.load("svm.xml"); // loading

	//Taking input image
	Mat test_img = imread(pImage, 0);
	resize(test_img, test_img, Size(w, h));
	test_img = test_img.reshape(0, 1);
	//imshow("shit_image", test_img);
	test_img.convertTo(test_img, CV_32FC1);

	//Predicting emotion
	float res = svm.predict(test_img);
	return res;
} //end of function setsvm()