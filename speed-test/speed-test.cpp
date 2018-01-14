#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase


#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "boost/filesystem.hpp"
#include <ctime>
#include <stdlib.h>

using namespace DBoW2;
using namespace DUtils;
using namespace std;

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
vector<cv::Mat> getFeatures(cv::Mat &im, cv::Ptr<cv::ORB> &orb);

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
	//out.resize(plain.rows);

  	for(int i = 0; i < plain.rows; ++i)
  		out[i] = plain.row(i);
}

vector<cv::Mat> getFeatures(cv::Mat &im, cv::Ptr<cv::ORB> &orb)
{
	cv::Mat mask;
	vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	orb->detectAndCompute(im, mask, keypoints, descriptors);
	
	vector<cv::Mat> features(descriptors.rows);
	changeStructure(descriptors, features);	
	return features;
}


int main(int argc, char** argv)
{

	if (argc < 3)
	{
		std::cout << "Usage:\n\tspeed-test <mem dir> <live dir> <(optional) GPU_ID (default is use CPU)> \n";
		return -1;
	}
	boost::filesystem::path mem(argv[1]);
	boost::filesystem::path live(argv[2]);
	boost::filesystem::directory_iterator end_itr; // NULL
	cv::Mat im;
	clock_t start;
	std::list<double> comp_t_list /* descriptor compute + database add times */, query_t_list /* descriptor compute + database query times */;
	double dt;


	// branching factor and depth levels 
	const int k = 9;
	const int L = 3;
	const WeightingType weight = TF_IDF;
	const ScoringType score = L1_NORM;

	std::cout << "Loading ORB Vocabulary ...\n";
	start = clock();
	OrbVocabulary voc(k, L, weight, score);
  	voc.loadFromTextFile("ORBvoc.txt");
	std::cout << "Time to load vocab tree = " << (double)(clock()-start)/CLOCKS_PER_SEC << "\n";
	cv::Ptr<cv::ORB> orb = cv::ORB::create();
	OrbDatabase db(voc, false, 0); // false = do not use direct index
	vector<cv::Mat> feats;
	std::cout << "\n----------------- Database Creation -------------------------\n";

	for (boost::filesystem::directory_iterator itr1(mem); itr1!=end_itr; ++itr1)
	{	
		std::cout << "loading database image from: " << itr1->path().string() << "\n";
		im = cv::imread(itr1->path().string(), cv::IMREAD_GRAYSCALE);
		start = clock();
		feats = getFeatures(im, orb);
		int id = db.add(feats);
		dt = ((double)clock()-start)/CLOCKS_PER_SEC;
		std::cout << "Time from image to DB = " << dt*1000 << " ms\n";
		comp_t_list.push_front(dt);  
		std::cout << "Added Image " << id << " to database\n\n";
	}
	
	std::cout << "\n----------------- Database Query -------------------------\n";

	// Okay now we have a database of descriptors, lets time the querie
	QueryResults q; // This is unused since we only do a speed test
	for (boost::filesystem::directory_iterator itr2(live); itr2!=end_itr; ++itr2)
	{	
		std::cout << "loading query image from: " << itr2->path().string() << "\n";
		im = cv::imread(itr2->path().string(), cv::IMREAD_GRAYSCALE);
		feats = getFeatures(im, orb);
		start = clock();
		db.query(feats, q); // query(im, false) means return 1 result in q, and don't add im's descriptor to database afterwards
		dt = ((double)clock()-start)/CLOCKS_PER_SEC;
		query_t_list.push_front(dt);  
		std::cout << q << "\n";
		std::cout << "Database query time = " << dt*1000 << " ms\n\n";
	}

 	std::cout << "\n\n\t\t\tResults\n\t\t\t-------\n\n";
        std::cout << "Time from image to DB:\n\t\t\t";
        double mean = std::accumulate(comp_t_list.begin(), comp_t_list.end(), 0.0) / (double)comp_t_list.size();

        double stdev = 0.0;
        for (double x : comp_t_list)
                stdev += pow(x-mean, 2);
        stdev = sqrt( stdev / ((double)comp_t_list.size()-1.0) );
        std::cout << "Mean: " << mean*1000 << ", StDev: " << stdev*1000 << " (ms) \n";
        
        std::cout << "Database querying time for a database of size " << query_t_list.size() << ":\n\t\t\t";
        mean = std::accumulate(query_t_list.begin(), query_t_list.end(), 0.0) / (double)query_t_list.size();
        stdev = 0.0;
        for (double x : query_t_list)
                stdev += pow(x-mean, 2);
        stdev = sqrt( stdev / ((double)query_t_list.size()-1.0) );
        std::cout << "Mean: " << mean*1000 << ", StDev: " << stdev*1000 << " (ms) \n";

} 


