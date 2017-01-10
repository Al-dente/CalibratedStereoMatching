#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cvaux.h>
#include <iostream>
#include <fstream>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

using namespace std;
using namespace cv;
using namespace boost::program_options;

//#define RIGHT		0 // コマンドライン引数で指定可能
//#define LEFT1		1
//#define LEFT2		2
//#define LEFT3		3
#define BOARD_W		7
#define BOARD_H		10
#define SCALE		20
#define ESC		1048603
#define SPACE		1048608

const Size BOARD_SIZE = Size(BOARD_W, BOARD_H);
const int N_CORNERS = BOARD_W * BOARD_H;

/* 深さマップの書き出し */
bool writeTxt (const string file_name, const Mat_<int> & mat) {
	string line;
	ofstream ofs(file_name.c_str());

	if(!ofs) {
		cout << boost::format("Cannot Open %s\n") % file_name;
		return false;
	}
	
	for(int j = 0; j < mat.rows; j++) {
		for(int i = 0; i < mat.cols; i++) {
			ofs << 255 - mat(j, i);
			if(i < mat.cols - 1) ofs << ",";
			if(i == mat.cols - 1) ofs << "\n";
		}
	}

	return true;
}

bool fexist(const char *filename) {
	FILE *fp;

	if((fp = fopen(filename, "r")) == NULL)
		return false;
	fclose(fp);
	return true;
}

bool setcamMatrixanddisCoeffs(const string file_name, Mat & cameraMatrix, Mat & distCoeffs) {
	FileStorage fs(file_name + ".xml", FileStorage::READ);
	if(!fs.isOpened()) {
		cout << "Cannot Open " << file_name << ".xml." << endl;
		return false;
	}

	FileNode node(fs.fs, NULL);
	read(node["cameraMatrix"], cameraMatrix);
	read(node["distCoeffs"], distCoeffs);
	cout << "cameraMatrix:" << endl;
	cout << cameraMatrix << endl;
	cout << "distCoeffs:" << endl;
	cout << distCoeffs << endl;
	fs.release();

	return true;
}

bool setmaps(const string file_name, Mat & map1x, Mat & map1y, Mat & map2x, Mat & map2y) {
	FileStorage fs(file_name + ".xml", FileStorage::READ);
	if(!fs.isOpened()) {
		cout << "Cannot Open " << file_name << ".xml." << endl;
		return false;
	}

	FileNode node(fs.fs, NULL);
	read(node["map1x"], map1x);
	read(node["map1y"], map1y);
	read(node["map2x"], map2x);
	read(node["map2y"], map2y);
	fs.release();
	return true;
}


/* カメラの設定 */
bool openCamera(VideoCapture cap, int width, int height) {
	if(!cap.isOpened()) {
		cout << "Cannot Open Camera." << endl;
		return false;
	}

	cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	cap.set(CV_CAP_PROP_FPS, 30.0);

	return true;
}

/* カメラからの画像を回転 */
/* void rotateFrame(Mat origF, Mat & rotatedF) {
 * 	float angle = 90.0, scale = 1.0;
 *	Point2f center(origF.cols*0.5, origF.rows*0.5);
 *	//cout << center << endl;	// Debug
 *	const Mat affine_matrix = getRotationMatrix2D(center, angle, scale);
 *	warpAffine(origF, rotatedF, affine_matrix, rotatedF.size());
 *}
 */

/* カメラからの画像を回転 */
/* (フレームの大きさに合わせるためにこちらの回転法を利用) */
void rotateFrame2(Mat origF, Mat & rotatedF) {
	Point2f src_pt[] = {Point2f(0, 0), Point2f(0, origF.rows), Point2f(origF.cols, 0)};
	Point2f dst_pt[] = {Point2f(0, origF.cols), Point2f(origF.rows, origF.cols), Point2f(0, 0)};
	Mat affine_matrix = getAffineTransform(src_pt, dst_pt);
	warpAffine(origF, rotatedF, affine_matrix, rotatedF.size());
}

/* 左端のノイズ対策  */
void regulateFrame(Mat & origF, int noise) {
	Point2f src_pt[] = {Point2f(0, 0), Point2f(0, origF.rows), Point2f(origF.cols, origF.rows)};
	Point2f dst_pt[] = {Point2f(noise, 0), Point2f(noise, origF.rows), Point2f(origF.cols + noise, origF.rows)};
	Mat affine_matrix = getAffineTransform(src_pt, dst_pt);
	warpAffine(origF, origF, affine_matrix, origF.size());
}

/* 各カメラのキャリブレーションに必要な画像を用意 */
bool CreateCalibSrcImgs(int cam_id, string file_name, int n_images, int width, int height) {
	VideoCapture cap(cam_id);
	if(!openCamera(cap, width, height)) return false;

	Mat frame;
	Mat r_frame(width, height, frame.type());
	int file_no = 0;
	namedWindow("Cap" + file_name, CV_WINDOW_AUTOSIZE);
	
	while(1) {
		cap >> frame;
		rotateFrame2(frame, r_frame);
		imshow("Cap" + file_name, r_frame);

		int k = waitKey(30);
		if(k == SPACE) {
			imwrite("SrcImg" + file_name  + "_" + to_string(file_no) + ".png", r_frame);
			cout << "Saved SrcImg" << file_name  << "_" << to_string(file_no) << ".png." << endl;
			file_no++;
		} else if(k == ESC) {
			return false;
		}

		if(file_no == n_images)
			break;
	}
	
	destroyAllWindows();
	return true;
}

/* カメラ間でのキャリブレーションに必要な画像を用意 */
bool CreateCalibSrcImgs2(int cam_right, int cam_left, string file_name, int n_images, int width, int height) {
	VideoCapture capR(cam_right);
	VideoCapture capL(cam_left);

	if(!openCamera(capR, width, height)) return false;
	if(!openCamera(capL, width, height)) return false;

	Mat frameR, frameL;
	Mat r_frameR(width, height, frameR.type()), r_frameL(width, height, frameL.type());
	int file_no = 0;
	namedWindow("Cap Right", CV_WINDOW_AUTOSIZE);
	namedWindow("Cap Left", CV_WINDOW_AUTOSIZE);

	while(1) {
		capR >> frameR;
		capL >> frameL;

		rotateFrame2(frameR, r_frameR);
		rotateFrame2(frameL, r_frameL);

		imshow("Cap Right", r_frameR);
		imshow("Cap Left", r_frameL);

		int k = waitKey(30);
		if(k == SPACE) {
			imwrite("SrcImg" + file_name + "_R_" + to_string(file_no) + ".png", r_frameR);
			cout << "Saved SrcImg" << file_name << "_R_" << to_string(file_no) << ".png." << endl;
			imwrite("SrcImg" + file_name + "_L_" + to_string(file_no) + ".png", r_frameL);
			cout << "Saved SrcImg" << file_name << "_L_" << to_string(file_no) << ".png." << endl;
			file_no++;
		} else if(k == ESC) {
			return false;
		}

		if(file_no == n_images)
			break;
	}

	destroyAllWindows();
	return true;
}

/* カメラ毎のキャリブレーションを行い,cameraMatrixとdistCoeffsを保存する. */
bool camera_calib(int camera_no, string camera_name, int n_images, int image_w, int image_h) {
	Size image_size = Size(image_h, image_w);
	
	if(!CreateCalibSrcImgs(camera_no, camera_name, n_images, image_w, image_h)) return false;
	
	vector<Mat> src_image(n_images);
	for(int i = 0; i < n_images; i++) {
		src_image[i] = imread("SrcImg" + camera_name +  "_" + to_string(i) + ".png");
		cout << "Read SrcImg" << camera_name << "_" << to_string(i) << ".png" << endl;
	}

	vector<vector<Point2f>> imagePoints;
	vector<Point2f> imageCorners;
	Mat gray_image;
	bool found;
	Mat dst_image;

	for(int i = 0; i < n_images; i++) {
		found = findChessboardCorners(src_image[i], BOARD_SIZE, imageCorners);
		
		if(found) {
			cvtColor(src_image[i], gray_image, CV_BGR2GRAY);
			cornerSubPix(gray_image, imageCorners, Size(9, 9), Size(-1, 1), TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.1));

			dst_image = src_image[i].clone();
			drawChessboardCorners(dst_image, BOARD_SIZE, imageCorners, found);
			imwrite("DstImg" + camera_name + "_" + to_string(i) + ".png", dst_image);
			
			for(int j = 0; j < n_images; j++)
				cout << j << " " << (int)(imageCorners[j].x + 0.5) << " " << (int)(imageCorners[j].y + 0.5) << endl;
		
		} else {
			cout << "Error: Cannot Find Chessboard Corners in SrcImg" << camera_name << "_" << i << ".png!" << endl;
			cout << "Please Re-prepare Source Images." << endl;
			return false;
		}

		imagePoints.push_back(imageCorners);
	}
	
	vector<vector<Point3f>> objectPoints;
	vector<Point3f> objectCorners;
	for(int j = 0; j < BOARD_H; j++) {
		for(int i = 0; i < BOARD_W; i++) {
			objectCorners.push_back(Point3f(i*SCALE, j*SCALE, 0.0f));
		}
	}
	for(int i = 0; i < n_images; i++)
		objectPoints.push_back(objectCorners);
	
	Mat cameraMatrix, distCoeffs;
	vector<Mat> rvecs, tvecs;
	double rms = calibrateCamera(objectPoints, imagePoints, image_size, cameraMatrix, distCoeffs, rvecs, tvecs);
	
	FileStorage fs(camera_name + ".xml", FileStorage::WRITE);
	fs << "cameraMatrix" << cameraMatrix;
	fs << "distCoeffs" << distCoeffs;
	fs << "rvec" << rvecs;
	fs << "tvec" << tvecs;
	fs.release();

	return true;
}

bool system_calib(int camera_noR, int camera_noL, string camera_nameR, string camera_nameL, int n_images, int image_w, int image_h) {
	string pair_name = camera_nameR + camera_nameL;
	Size image_size = Size(image_h, image_w);
	
	vector<Mat> src_imageR(n_images), src_imageL(n_images);	// 元画像を格納するベクトル
	for(int i = 0; i < n_images; i++) {
		src_imageR[i] = imread("SrcImg" + pair_name + "_R_" + to_string(i) + ".png");
		cout << "Read SrcImg" << pair_name << "_R_" << to_string(i) << ".png." << endl;
		src_imageL[i] = imread("SrcImg" + pair_name + "_L_" + to_string(i) + ".png");
		cout << "Read SrcImg" << pair_name << "_L_" << to_string(i) << ".png." << endl;
	}
	
	Mat cameraMatrixR, cameraMatrixL;// 各カメラのカメラ行列
	Mat distCoeffsR, distCoeffsL;;		// 各カメラの歪み係数
	
	/* 各カメラのカメラ行列と歪み係数を取得する. */
	if(!setcamMatrixanddisCoeffs(camera_nameR, cameraMatrixR, distCoeffsR)) return false;
	if(!setcamMatrixanddisCoeffs(camera_nameL, cameraMatrixL, distCoeffsL)) return false;
	
	/* 各カメラの元画像を歪み補正 */
	vector<Mat> undist_imageR(n_images), undist_imageL(n_images);	// 補正画像を格納するベクトル
	for(int i = 0; i < n_images; i++) {
		undistort(src_imageR[i], undist_imageR[i], cameraMatrixR, distCoeffsR);
		imwrite("undist_image_" + camera_nameR + "_" + to_string(i) + ".png", undist_imageR[i]);
		undistort(src_imageL[i], undist_imageL[i], cameraMatrixL, distCoeffsL);
		imwrite("undist_image_" + camera_nameL + "_" + to_string(i) + ".png", undist_imageL[i]);
	}

	/*  */
	vector<vector<Point2f>> imagePointsR, imagePointsL;
	vector<Point2f> imageCornersR, imageCornersL;
	bool foundR, foundL;
	Mat gray_imageR, gray_imageL;
	Mat check_imageR, check_imageL;
	for(int i = 0; i < n_images; i++) {
		foundR = findChessboardCorners(undist_imageR[i], BOARD_SIZE, imageCornersR, CV_CALIB_CB_ADAPTIVE_THRESH + CV_CALIB_CB_FILTER_QUADS);
		foundL = findChessboardCorners(undist_imageL[i], BOARD_SIZE, imageCornersL, CV_CALIB_CB_ADAPTIVE_THRESH + CV_CALIB_CB_FILTER_QUADS);
		if(foundR) {	
			cvtColor(undist_imageR[i], gray_imageR, CV_BGR2GRAY);
			cornerSubPix(gray_imageR, imageCornersR, Size(9, 9), Size(-1, 1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			check_imageR = undist_imageR[i].clone();
			drawChessboardCorners(check_imageR, BOARD_SIZE, imageCornersR, foundR);
			imwrite("check_image_" + camera_nameR + "_" + to_string(i) + ".png", check_imageR);
		} else {
			cout << "Error: Cannnot Find Chessboard Corners in SrcImg" << pair_name << "_R_" << to_string(i) << ".png!" << endl;
			cout << "Please Re-prepare Source Images." << endl;
			return false;	
		}
		if(foundL) {
			cvtColor(undist_imageL[i], gray_imageL, CV_BGR2GRAY);
			cornerSubPix(gray_imageL, imageCornersL, Size(9, 9), Size(-1, 1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			check_imageL = undist_imageL[i].clone();
  			drawChessboardCorners(check_imageL, BOARD_SIZE, imageCornersL, foundL);
			imwrite("check_image_" + camera_nameL + "_" + to_string(i) + ".png", check_imageL);
		} else {
			cout << "Error: Cannnot Find Chessboard Corners in SrcImg" << pair_name << "_L_" << to_string(i) << ".png!" << endl;
			cout << "Please Re-prepare Source Images." << endl;
			return false;
		}

		if(foundR && foundL) {
			imagePointsR.push_back(imageCornersR);
			imagePointsL.push_back(imageCornersL);
		}
	}

	vector<vector<Point3f>> objectPoints;
	vector<Point3f> objectCorners;
	for(int i = 0; i < N_CORNERS; i++)
		objectCorners.push_back(Point3f(i / BOARD_W, i % BOARD_W, 0.0f));

	for(int i= 0; i < n_images; i++) 
		objectPoints.push_back(objectCorners);

	cout << "Starting Calibration." << endl;
	Mat cameraMatrix1, cameraMatrix2;
	Mat distCoeffs1, distCoeffs2;
	Mat R, T, E, F;

	stereoCalibrate(objectPoints, imagePointsR, imagePointsL, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size, R, T, E, F, cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5), CV_CALIB_SAME_FOCAL_LENGTH + CV_CALIB_ZERO_TANGENT_DIST);
	
	FileStorage fs(pair_name + ".xml", FileStorage::WRITE);
	fs << "cameraMatrix1" << cameraMatrix1;
	fs << "cameraMatrix2" << cameraMatrix2;
	fs << "distCoeffs1" << distCoeffs1;
	fs << "distCoeffs2" << distCoeffs2;
	fs << "R" << R;
	fs << "T" << T;
	fs << "E" << E;
	fs << "F" << F;

	cout << "Completed Calibration." << endl;
	cout << "Starting Rectification." << endl;

	Mat R1, R2, P1, P2, Q;
	stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size, R, T, R1, R2, P1, P2, Q);
	
	fs << "R1" << R1;
	fs << "R2" << R2;
	fs << "P1" << P1;
	fs << "P2" << P2;
	fs << "Q" << Q;

	cout << "Completed Rectification." << endl;
	cout << "Applying Undistort." << endl;

	Mat map1x, map1y, map2x, map2y;
	initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, image_size, CV_32FC1, map1x, map1y);
	initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, image_size, CV_32FC1, map2x, map2y);
	
	fs << "map1x" << map1x;
	fs << "map1y" << map1y;
	fs << "map2x" << map2x;
	fs << "map2y" << map2y;
	
	cout << "Undistort Complete." << endl;

	Mat calibrated_imageR, calibrated_imageL;
	for(int i = 0; i < n_images; i++) {
		check_imageR = undist_imageR[i];
		remap(check_imageR, calibrated_imageR, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
		check_imageL = undist_imageL[i];
		remap(check_imageL, calibrated_imageL, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
		
		imwrite("calibrated_image_" + camera_nameR + "_" + to_string(i) + ".png", calibrated_imageR);
		imwrite("calibrated_image_" + camera_nameL + "_" + to_string(i) + ".png", calibrated_imageL);
	}

	fs.release();
	 	
	return true;
}

/* Main関数 */
int main(int argc, const char* argv[]) {
	/* コマンドラインオプションの定義 */
	options_description options0("一般的なオプション");
	options_description options1("カメラの指定に関するオプション");
	options_description options2("カメラからの取得動画に関するオプション");
	options_description options3("キャリブレーションに関するオプション");
	options_description options4("ステレオマッチングに関するオプション");
	
	options0.add_options()
		("help,h", 						"ヘルプを表示")
	;

	options1.add_options()
		("r_cam,R",	value<string>()->default_value("R1"),	"右カメラを指定(R1)")
		("l_cam,L",	value<string>()->default_value("L1"),	"左カメラを指定(L1-3)")
	;

	options2.add_options()
		("cap_w,W",	value<int>()->default_value(320),	"取得動画の横ピクセルを指定")
		("cap_h,H",	value<int>()->default_value(240),	"取得動画の縦ピクセルを指定")
	;

	options3.add_options()
		("camera_calib,C",					"カメラ毎のキャリブレーションを行う")
		("system_calib,S",					"カメラ間のキャリブレーションを行う")
		("n_images,N",	value<int>()->default_value(10),	"キャリブレーションに利用する画像の枚数を指定")
	;

	options4.add_options()
		("threshold,T",	value<int>()->default_value(100),	"マスク画像作成の際の閾値を指定")
	;

	options0.add(options1).add(options2).add(options3).add(options4);
	variables_map vm;

	try {
		store(parse_command_line(argc, argv, options0), vm);
	} catch(exception &e) {
		cout << e.what() << endl;
		return -1;	
	}

	notify(vm);

	if(vm.count("help")) {
		cout << options0 << endl;
		return 0;
	}

	string R_CAM, L_CAM;
	int RIGHT, LEFT;

	if(vm.count("r_cam")) {
		R_CAM = vm["r_cam"].as<string>();
		if(R_CAM == "R1") {
			RIGHT = 0;
		} else {
			cout << "Please Input Correct Right Camera." << endl;
			return -1;
		}
	} else {
		R_CAM = "R1";
		RIGHT = 0;
	}

	if(vm.count("l_cam")) {
		L_CAM = vm["l_cam"].as<string>();
		if(L_CAM == "L1") {
			LEFT = 1;
		} else if(L_CAM == "L2") {
			LEFT = 2;
		} else if(L_CAM == "L3") {
			LEFT = 3;
		} else {
			cout << "Please Input Correct Left Camera." << endl;
			return -1;
		}
	} else {
		L_CAM = "L1";
		LEFT = 1;
	}

	const int WIDTH = vm["cap_w"].as<int>();
	const int HEIGHT = vm["cap_h"].as<int>();	// ex) 320x240: WIDTH=320, HEIGHT=240
	const int N_IMAGES = vm["n_images"].as<int>();
	const Size IMAGE_SIZE = Size(HEIGHT, WIDTH);
	const string PAIR_NAME = R_CAM + L_CAM;
	const int NOD = 64;
	const int THRESHOLD = vm["threshold"].as<int>();

	/* キャリブレーション(各カメラ) */
	if(vm.count("camera_calib")) {
		cout << "Start Camera Calibration." << endl;
		if(!fexist("R1.xml"))
			if(!camera_calib(RIGHT, R_CAM, N_IMAGES, WIDTH, HEIGHT)) return -1;
		if(!camera_calib(LEFT, L_CAM, N_IMAGES, WIDTH, HEIGHT)) return -1;
		cout << "Complete Camera Calibration." << endl;
	}
	
	/* キャリブレーション(システム) */
	if(vm.count("system_calib")) {
		cout << "Prepare Images of System Calibration." << endl;
		if(!CreateCalibSrcImgs2(RIGHT, LEFT, PAIR_NAME, N_IMAGES, WIDTH, HEIGHT))	return -1;
		cout << "Prepared Images of System Calibration." << endl;
	}

	cout << "Start System Calibration." << endl;
	if(!system_calib(RIGHT, LEFT, R_CAM, L_CAM, N_IMAGES, WIDTH, HEIGHT))	return -1;
	cout << "Complete System Calibration." << endl;	
	

	Mat map1x, map1y, map2x, map2y;
	setmaps(PAIR_NAME, map1x, map1y, map2x, map2y);

 	VideoCapture capR1(RIGHT);
	VideoCapture capL1(LEFT);
//	VideoCapture capL2(LEFT2);
//	VideoCapture capL3(LEFT3);

	if(!openCamera(capR1, WIDTH, HEIGHT)) return -1;
	if(!openCamera(capL1, WIDTH, HEIGHT)) return -1;
//	if(!openCamera(capL2, WIDTH, HEIGHT)) return -1;
//	if(!openCamera(capL3, WIDTH, HEIGHT)) return -1;

	//namedWindow("Cap Right", CV_WINDOW_NORMAL);
	//namedWindow("Cap Left", CV_WINDOW_NORMAL);
	namedWindow("Dis", CV_WINDOW_NORMAL);
	namedWindow("Out", CV_WINDOW_NORMAL);
	namedWindow("Right", CV_WINDOW_NORMAL);
	namedWindow("Left1", CV_WINDOW_NORMAL);
//	namedWindow("Rotation Left2", CV_WINDOW_NORMAL);
//	namedWindow("Rotation Left3", CV_WINDOW_NORMAL);

	Mat frameR1, frameL1, frameL2, frameL3;
	Mat r_frameR1(WIDTH, HEIGHT, frameR1.type()), r_frameL1(WIDTH, HEIGHT, frameL1.type()),	r_frameL2(WIDTH, HEIGHT, frameL2.type()), r_frameL3(WIDTH, WIDTH, frameL3.type());	
	Mat f_frameR1(WIDTH, HEIGHT, frameR1.type()), f_frameL1(WIDTH, HEIGHT, frameL1.type()),	f_frameL2(WIDTH, HEIGHT, frameL2.type()), f_frameL3(WIDTH, WIDTH, frameL3.type());		
	Mat frame(WIDTH, WIDTH, frameR1.type());
	Mat dis, out, mask;

	int n = 0;
	while(1) {
		capR1 >> frameR1;
		capL1 >> frameL1;
//		capL2 >> frameL2;
//		capL3 >> frameL3;

		if(frameR1.empty() || frameL1.empty()) break;

		rotateFrame2(frameR1, r_frameR1);
		rotateFrame2(frameL1, r_frameL1);
//		rotateFrame2(frameL2, r_frameL2);
//		rotateFrame2(frameL3, r_frameL3);

		/* キャリブレーション実行 */	
		remap(r_frameR1, f_frameR1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
		remap(r_frameL1, f_frameL1, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT, Scalar());
		
		/* 左端のノイズ対策 */
		//regulateFrame(f_frameR1, NOD);
		//regulateFrame(f_frameL1, NOD);
		
		/* StereoSGBM開始 */
		StereoSGBM sgbm;
		int cn = frameL1.channels();
		
		/*----------パラメータ----------*/
		sgbm.minDisparity = 0;		// 取り得る最小の視差値
		sgbm.numberOfDisparities = NOD;	// 取り得る最大の視差値
		sgbm.SADWindowSize = 3;		// マッチングされるブロックのサイズ(3-11)
		sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
			// 視差のなめらかさを制御するパラメータ、この値が大きくなる視差がなめらかになる
		sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
			// 
		sgbm.disp12MaxDiff = 10;	// left-right視差チェックにおいて許容される最大の差(整数ピクセル単位)
		sgbm.preFilterCap = 63;		// 事前フィルタにおいて画像ピクセルを切り捨てる閾値
		sgbm.uniquenessRatio = 30;	// パーセント単位で表現されるマージン
		sgbm.speckleWindowSize = 300;	// ノイズスペックや無効なピクセルが考慮されたなめらかな視差領域の最大サイズ(0or50-200)
		sgbm.speckleRange = 1;		// それぞれの連結成分における最大視差値(16 or 32)
		sgbm.fullDP = false;		// 完全な2パス
		/*----------パラメータ----------*/

		sgbm(f_frameL1, f_frameR1, frame);

		frame.convertTo(dis, CV_8U, 255/((55 & -16)*16.));
		threshold(dis, mask, THRESHOLD, 255, THRESH_BINARY);
		out = Mat::zeros(3, 3, CV_32FC1);
		f_frameL1.copyTo(out, mask);
		
		//imshow("Cap Right", frameR1);
		//imshow("Cap Left1", frameL1);
		//imshow("Cap Left2", frameL2);
		//imshow("Cap Left3", frameL3);	
		imshow("Dis", dis);
		imshow("Out", out);
		imshow("Right", f_frameR1);
		imshow("Left1", f_frameL1);
//		imshow("Rotation Left2", r_frameL2);
//		imshow("Rotation Left3", r_frameL3);

		/* 「cvMoveWindow」はCの関数*/
		//cvMoveWindow("Cap Right", 0, 100);
		//cvMoveWindow("Cap Left1", 320, 100);
		//cvMoveWindow("Cap Left2", 640, 100);
		//cvMoveWindow("Cap Left3", 960, 100);
		cvMoveWindow("Right", 0, 340);
		cvMoveWindow("Left1", 320, 340);
//		cvMoveWindow("Rotation Left2", 640, 340);
//		cvMoveWindow("Rotation Left3", 960, 340);
		cvMoveWindow("Dis", 0, 580);
		cvMoveWindow("Out", 320, 580);
		
		int k = waitKey(30);
		if(k == SPACE) {
			imwrite(PAIR_NAME + "_" + to_string(WIDTH) + "*" + to_string(HEIGHT) + "_" + to_string(THRESHOLD) + "_Right_img" + to_string(n) + ".png", f_frameR1);
			imwrite(PAIR_NAME + "_" + to_string(WIDTH) + "*" + to_string(HEIGHT) + "_" + to_string(THRESHOLD) + "_Left_img" + to_string(n) + ".png", f_frameL1);
			imwrite(PAIR_NAME + "_" + to_string(WIDTH) + "*" + to_string(HEIGHT) + "_" + to_string(THRESHOLD) + "_Dis_img" + to_string(n) + ".png", dis);
			imwrite(PAIR_NAME + "_" + to_string(WIDTH) + "*" + to_string(HEIGHT) + "_" + to_string(THRESHOLD) + "_Mask_img" + to_string(n) + ".png", out);
			cout << "Saved images" << endl;
			n++;
		} else if(k >= 0) {
			writeTxt("dis.csv", dis);
			break;
		}
	}	
	return 0;		
}

